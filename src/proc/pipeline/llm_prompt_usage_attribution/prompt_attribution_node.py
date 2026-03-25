from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from dspy.adapters import ChatAdapter
from returns.pipeline import is_successful
from returns.result import Failure, Result, Success

import dspy

from proc.base.dspy_llm import DSpyLLM
from proc.base.proc_error import ProcError
from proc.base.proc_score import ProcScore, ProcScoreContext
from proc.base.timing import timed
from proc.pipeline.dataset.base_dataset import BaseDataset
from proc.pipeline.llm_prompt_usage_attribution._hf_attribution_base import (
    HFAttributionBase,
    _HF_DEFAULT_CHART_SCORE_THRESHOLD,
    _HF_DEFAULT_INTERNAL_BATCH,
    _HF_DEFAULT_MODEL,
)
from proc.pipeline.output_result_auditor.score_extractor import INVALID_SCORE, ScoreExtractor

_DEFAULT_MAX_NEW_TOKENS: int = 12
_DEFAULT_IG_STEPS: int = 24
_CONTROL_TOKENS: frozenset[str] = frozenset({
    '<|im_start|>', '<|im_end|>', 'system', 'user', 'assistant',
})


class TokenRegion(StrEnum):
    CONTROL = 'control'
    INSTRUCTION = 'instruction'
    EMAIL = 'email'
    OTHER = 'other'


@dataclass(slots=True)
class RankedToken:
    index: int
    token: str
    region: TokenRegion
    score: float
    normalized_abs_score: float


@dataclass(slots=True)
class StepAttribution:
    step_index: int
    target_token: str
    target_token_id: int
    logprob: float
    prompt_token_scores: list[float]


@dataclass(slots=True)
class ExampleAttributionReport:
    example_index: int
    prompt_text: str
    generated_text: str
    target_text: str
    used_generated_target: bool
    prompt_tokens: list[str]
    token_regions: list[TokenRegion]
    aggregate_scores: list[float]
    steps: list[StepAttribution]
    top_positive_tokens: list[RankedToken]
    top_negative_tokens: list[RankedToken]
    top_positive_email_tokens: list[RankedToken]
    top_negative_email_tokens: list[RankedToken]
    instruction_score_sum: float
    instruction_abs_score_sum: float
    email_score_sum: float
    email_abs_score_sum: float
    output_dir: str


@dataclass(slots=True)
class PromptAttributionSummary(ProcScoreContext):
    hf_model_name: str
    output_dir: str
    example_count: int
    report_files: list[str]


class PromptAttributionNode(HFAttributionBase):
    """
    Prompt attribution auditor.

    Design goals:
      - Captum-based sequence attribution for the actual generated answer or a forced target
      - Captum-style artifacts: per-example prompt heatmap, step x prompt matrix, HTML report, JSON dump
    """

    _SCORE = 1.0

    def __init__(
        self,
        dataset: BaseDataset,
        llm: DSpyLLM[Any, Any],
        scorer: ScoreExtractor,
        hf_model_name: str = _HF_DEFAULT_MODEL,
        output_dir: str | Path = 'runs/prompt_attribution',
        generation_max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        ig_steps: int = _DEFAULT_IG_STEPS,
        internal_batch_size: int = _HF_DEFAULT_INTERNAL_BATCH,
        attr_device: str = 'cpu',
        force_dtype: torch.dtype | None = None,
        target_text: str | None = None,
        save_html: bool = True,
        save_plots: bool = True,
        top_k_tokens: int = 12,
        truncate_to_model_max_length: bool = True,
        chart_score_threshold: float = _HF_DEFAULT_CHART_SCORE_THRESHOLD,
        peft_model_id: str | None = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            llm=llm,
            scorer=scorer,
            hf_model_name=hf_model_name,
            internal_batch_size=internal_batch_size,
            attr_device=attr_device,
            force_dtype=force_dtype,
            output_dir=output_dir,
            save_html=save_html,
            save_plots=save_plots,
            chart_score_threshold=chart_score_threshold,
            peft_model_id=peft_model_id,
        )
        self._generation_max_new_tokens = generation_max_new_tokens
        self._ig_steps = ig_steps
        self._target_text = target_text
        self._top_k_tokens = top_k_tokens
        self._truncate_to_model_max_length = truncate_to_model_max_length

    def invoke(self) -> Result[ProcScore, ProcError]:
        predictor_result = self._get_predictor()
        if not is_successful(predictor_result):
            return predictor_result
        predictor = predictor_result.unwrap()
        adapter = ChatAdapter()

        load_result = self._load_model()
        if not is_successful(load_result):
            return Failure(load_result.failure())
        model, tokenizer = load_result.unwrap()
        model.to(self._device)
        embed_layer = model.get_input_embeddings()
        report_files: list[str] = []
        example_count = 0

        for index, example in enumerate(self._dataset.load()):
            example_count += 1
            self._logger.info('PromptAttributionNode: processing example[%d]', index)
            with dspy.context(cache=False):
                pred = self._llm(**example.inputs())
            score = self._scorer.extraction_metric(example, pred)
            if score == INVALID_SCORE:
                return Failure(ProcError(f'Cannot score example[{index}]'))

            prompt_text = self._render_prompt(adapter, predictor, example, tokenizer)
            enc = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False)
            input_ids = enc['input_ids']
            if self._truncate_to_model_max_length and getattr(model.config, 'max_position_embeddings', None):
                max_len = int(model.config.max_position_embeddings)
                if input_ids.shape[1] > max_len:
                    self._logger.warning(
                        'PromptAttributionNode: truncating prompt from %d to %d tokens',
                        input_ids.shape[1],
                        max_len,
                    )
                    input_ids = input_ids[:, -max_len:]

            report = self._attribute_example(
                example_index=index,
                model=model,
                tokenizer=tokenizer,
                embed_layer=embed_layer,
                input_ids=input_ids,
                prompt_text=prompt_text,
            )
            example_dir = self._output_dir / f'example_{index:03d}'
            example_dir.mkdir(parents=True, exist_ok=True)
            paths = self._save_report_artifacts(example_dir, report)
            report_files.extend(paths)
            self._pretty_log_report(report)

        summary = PromptAttributionSummary(
            hf_model_name=self._hf_model_name,
            output_dir=str(self._output_dir),
            example_count=example_count,
            report_files=report_files,
        )
        with open(self._output_dir / 'summary.json', 'w', encoding='utf-8') as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)

        return Success(ProcScore(value=self._SCORE, context=summary))

    def _attribute_example(
        self,
        example_index: int,
        model: Any,
        tokenizer: Any,
        embed_layer: torch.nn.Module,
        input_ids: torch.Tensor,
        prompt_text: str,
    ) -> ExampleAttributionReport:
        input_ids = input_ids.to(self._device)
        generated_ids, generated_text = self._generate_answer(model, tokenizer, input_ids)

        used_generated_target = self._target_text is None
        target_text = generated_text if used_generated_target else self._target_text or ''
        target_ids = tokenizer(target_text, return_tensors='pt', add_special_tokens=False)['input_ids'].to(self._device)
        if target_ids.numel() == 0:
            raise ValueError('Target text tokenized to an empty sequence.')

        prompt_embeds = embed_layer(input_ids).detach()
        baseline_embeds = torch.zeros_like(prompt_embeds)
        prompt_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        token_regions = self._infer_regions(prompt_tokens)

        step_results: list[StepAttribution] = []
        step_vectors: list[torch.Tensor] = []
        prefix_ids = torch.empty((1, 0), dtype=input_ids.dtype, device=self._device)

        for step_idx in range(target_ids.shape[1]):
            step_target_id = int(target_ids[0, step_idx].item())
            step_target_token = tokenizer.decode([step_target_id])
            prefix_embeds = embed_layer(prefix_ids).detach() if prefix_ids.shape[1] > 0 else None

            def forward_fn(prompt_embeds_batch: torch.Tensor) -> torch.Tensor:
                if prefix_embeds is not None:
                    prefix_batch = prefix_embeds.expand(prompt_embeds_batch.shape[0], -1, -1)
                    full_embeds = torch.cat([prompt_embeds_batch, prefix_batch], dim=1)
                else:
                    full_embeds = prompt_embeds_batch
                out = model(inputs_embeds=full_embeds)
                logits = out.logits[:, -1, :]
                log_probs = F.log_softmax(logits.float(), dim=-1)
                return log_probs[:, step_target_id]

            ig = IntegratedGradients(forward_fn)
            with timed(f'example[{example_index}] step[{step_idx}] ig.attribute', logger=self._logger):
                attributions, _ = ig.attribute(
                    inputs=prompt_embeds,
                    baselines=baseline_embeds,
                    n_steps=self._ig_steps,
                    internal_batch_size=self._internal_batch_size,
                    return_convergence_delta=True,
                )

            token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu()
            step_vectors.append(token_scores)

            with torch.no_grad():
                full_ids = torch.cat([input_ids, prefix_ids], dim=1)
                out = model(input_ids=full_ids)
                step_logprob = float(F.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0, step_target_id].item())

            step_results.append(StepAttribution(
                step_index=step_idx,
                target_token=step_target_token,
                target_token_id=step_target_id,
                logprob=step_logprob,
                prompt_token_scores=token_scores.tolist(),
            ))
            prefix_ids = torch.cat([prefix_ids, target_ids[:, step_idx:step_idx + 1]], dim=1)

        aggregate_scores_tensor = torch.stack(step_vectors, dim=0).sum(dim=0)
        aggregate_scores = aggregate_scores_tensor.tolist()
        rankings = self._build_rankings(prompt_tokens, token_regions, aggregate_scores)
        instruction_score_sum, instruction_abs_score_sum = self._region_sums(aggregate_scores, token_regions, TokenRegion.INSTRUCTION)
        email_score_sum, email_abs_score_sum = self._region_sums(aggregate_scores, token_regions, TokenRegion.EMAIL)

        return ExampleAttributionReport(
            example_index=example_index,
            prompt_text=prompt_text,
            generated_text=generated_text,
            target_text=target_text,
            used_generated_target=used_generated_target,
            prompt_tokens=prompt_tokens,
            token_regions=token_regions,
            aggregate_scores=aggregate_scores,
            steps=step_results,
            top_positive_tokens=rankings['positive_all'],
            top_negative_tokens=rankings['negative_all'],
            top_positive_email_tokens=rankings['positive_email'],
            top_negative_email_tokens=rankings['negative_email'],
            instruction_score_sum=instruction_score_sum,
            instruction_abs_score_sum=instruction_abs_score_sum,
            email_score_sum=email_score_sum,
            email_abs_score_sum=email_abs_score_sum,
            output_dir='',
        )

    def _save_report_artifacts(self, example_dir: Path, report: ExampleAttributionReport) -> list[str]:
        report.output_dir = str(example_dir)
        report_json = example_dir / 'report.json'
        with open(report_json, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, ensure_ascii=False, indent=2)
        files = [str(report_json)]

        if self._save_plots:
            prompt_plot = example_dir / 'prompt_heatmap.png'
            matrix_plot = example_dir / 'step_matrix.png'
            self._plot_prompt_heatmap(report, prompt_plot)
            self._plot_step_matrix(report, matrix_plot)
            files.extend([str(prompt_plot), str(matrix_plot)])

        if self._save_html:
            html_path = example_dir / 'report.html'
            self._write_html_report(report, html_path)
            files.append(str(html_path))

        return files

    def _pretty_log_report(self, report: ExampleAttributionReport) -> None:
        self._logger.info('=== Prompt ===\n%s', report.prompt_text)
        self._logger.info('=== Model generated answer ===\n%s', report.generated_text)
        self._logger.info('=== Attribution target ===\n%s\nused_generated_target=%s', report.target_text, report.used_generated_target)
        if report.top_positive_tokens:
            tok = report.top_positive_tokens[0]
            self._logger.info('=== Top POSITIVE non-control token ===\nindex=%d token=%r score=%+.6f', tok.index, tok.token, tok.score)
        if report.top_negative_tokens:
            tok = report.top_negative_tokens[0]
            self._logger.info('=== Top NEGATIVE non-control token ===\nindex=%d token=%r score=%+.6f', tok.index, tok.token, tok.score)
        if report.top_positive_email_tokens:
            tok = report.top_positive_email_tokens[0]
            self._logger.info('=== Top POSITIVE EMAIL token ===\nindex=%d token=%r score=%+.6f', tok.index, tok.token, tok.score)
        if report.top_negative_email_tokens:
            tok = report.top_negative_email_tokens[0]
            self._logger.info('=== Top NEGATIVE EMAIL token ===\nindex=%d token=%r score=%+.6f', tok.index, tok.token, tok.score)

    def _plot_prompt_heatmap(self, report: ExampleAttributionReport, out_path: Path) -> None:
        scores = torch.tensor(report.aggregate_scores, dtype=torch.float32)
        max_abs = float(scores.abs().max().item()) if scores.numel() else 1.0
        keep = self._apply_chart_threshold(scores)
        if not keep:
            self._logger.warning(f'_plot_prompt_heatmap: all tokens filtered by threshold={self._chart_score_threshold}, skipping')
            return
        filtered_scores = scores[keep].unsqueeze(0)
        filtered_labels = [self._display_token(report.prompt_tokens[i]) for i in keep]
        fig_w = max(12, min(0.35 * len(keep), 40))
        fig, ax = plt.subplots(figsize=(fig_w, 3.5))
        im = ax.imshow(filtered_scores.numpy(), aspect='auto', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
        ax.set_yticks([])
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels(filtered_labels, rotation=90, fontsize=8)
        ax.set_title(f'Prompt attribution — example {report.example_index} (threshold={self._chart_score_threshold})')
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    def _plot_step_matrix(self, report: ExampleAttributionReport, out_path: Path) -> None:
        if not report.steps:
            return
        matrix = torch.tensor([s.prompt_token_scores for s in report.steps], dtype=torch.float32)
        max_abs = float(matrix.abs().max().item()) if matrix.numel() else 1.0
        # Filter columns (prompt tokens) by their max absolute score across all steps
        col_max_abs = matrix.abs().max(dim=0).values
        keep = self._apply_chart_threshold(col_max_abs)
        if not keep:
            self._logger.warning(f'_plot_step_matrix: all tokens filtered by threshold={self._chart_score_threshold}, skipping')
            return
        filtered_matrix = matrix[:, keep]
        filtered_labels = [self._display_token(report.prompt_tokens[i]) for i in keep]
        fig_w = max(12, min(0.35 * len(keep), 40))
        fig_h = max(3, min(0.55 * len(report.steps) + 2, 16))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(filtered_matrix.numpy(), aspect='auto', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels(filtered_labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(report.steps)))
        ax.set_yticklabels([self._display_token(s.target_token) for s in report.steps], fontsize=9)
        ax.set_xlabel('Prompt token')
        ax.set_ylabel('Target token')
        ax.set_title(f'Target-step attribution matrix — example {report.example_index} (threshold={self._chart_score_threshold})')
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)

    def _write_html_report(self, report: ExampleAttributionReport, out_path: Path) -> None:
        max_abs = max((abs(x) for x in report.aggregate_scores), default=1.0) or 1.0
        prompt_html = ' '.join(
            self._html_span(tok, score, max_abs, region)
            for tok, score, region in zip(report.prompt_tokens, report.aggregate_scores, report.token_regions)
        )
        step_rows = '\n'.join(
            f'<tr><td>{step.step_index}</td><td>{html.escape(self._display_token(step.target_token))}</td><td>{step.target_token_id}</td><td>{step.logprob:.6f}</td></tr>'
            for step in report.steps
        )
        body = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>Prompt attribution report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
.token {{ padding: 2px 4px; margin: 1px; border-radius: 4px; display: inline-block; }}
.meta {{ margin-bottom: 16px; }}
table {{ border-collapse: collapse; margin-top: 16px; }}
td, th {{ border: 1px solid #ddd; padding: 6px 10px; }}
small {{ color: #555; }}
</style></head><body>
<h1>Prompt attribution — example {report.example_index}</h1>
<div class="meta">
<p><strong>Generated answer:</strong> {html.escape(report.generated_text)}</p>
<p><strong>Attribution target:</strong> {html.escape(report.target_text)} &nbsp; <small>used_generated_target={report.used_generated_target}</small></p>
<p><strong>Instruction score sum:</strong> {report.instruction_score_sum:+.6f} &nbsp; <strong>Email score sum:</strong> {report.email_score_sum:+.6f}</p>
</div>
<h2>Prompt tokens</h2>
<div>{prompt_html}</div>
<h2>Per-target-token log-probs</h2>
<table><thead><tr><th>Step</th><th>Token</th><th>Token ID</th><th>Logprob</th></tr></thead><tbody>{step_rows}</tbody></table>
</body></html>'''
        out_path.write_text(body, encoding='utf-8')

    def _html_span(self, token: str, score: float, max_abs: float, region: str) -> str:
        return self._make_html_span(self._display_token(token), score, max_abs, region)

    def _build_rankings(self, prompt_tokens: list[str], token_regions: list[TokenRegion], scores: list[float]) -> dict[str, list[RankedToken]]:
        max_abs = max((abs(x) for x in scores), default=1.0) or 1.0
        records: list[RankedToken] = []
        for idx, (tok, region, score) in enumerate(zip(prompt_tokens, token_regions, scores)):
            if self._is_control_like(tok):
                continue
            records.append(RankedToken(
                index=idx,
                token=tok,
                region=region,
                score=float(score),
                normalized_abs_score=float(abs(score) / max_abs),
            ))
        positive_all = sorted((r for r in records if r.score > 0), key=lambda r: r.score, reverse=True)[:self._top_k_tokens]
        negative_all = sorted((r for r in records if r.score < 0), key=lambda r: r.score)[:self._top_k_tokens]
        positive_email = sorted((r for r in records if r.region == TokenRegion.EMAIL and r.score > 0), key=lambda r: r.score, reverse=True)[:self._top_k_tokens]
        negative_email = sorted((r for r in records if r.region == TokenRegion.EMAIL and r.score < 0), key=lambda r: r.score)[:self._top_k_tokens]
        return {
            'positive_all': positive_all,
            'negative_all': negative_all,
            'positive_email': positive_email,
            'negative_email': negative_email,
        }

    def _region_sums(self, scores: list[float], regions: list[TokenRegion], name: TokenRegion) -> tuple[float, float]:
        vals = [float(s) for s, r in zip(scores, regions) if r == name]
        return sum(vals), sum(abs(v) for v in vals)

    def _generate_answer(self, model: Any, tokenizer: Any, input_ids: torch.Tensor) -> tuple[torch.Tensor, str]:
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=self._generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out_ids[:, input_ids.shape[1]:]
        text = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
        return new_ids, text

    def _infer_regions(self, prompt_tokens: list[str]) -> list[TokenRegion]:
        regions: list[TokenRegion] = []
        current: TokenRegion = TokenRegion.INSTRUCTION
        email_seen = False
        answer_seen = False
        for tok in prompt_tokens:
            disp = self._display_token(tok)
            if tok in _CONTROL_TOKENS or 'im_start' in tok or 'im_end' in tok or 'Ċ' in tok:
                regions.append(TokenRegion.CONTROL)
                continue
            if re.fullmatch(r'\.?\n\n', disp):
                regions.append(TokenRegion.CONTROL)
                continue
            if disp.strip() == 'Email':
                current = TokenRegion.INSTRUCTION
                email_seen = True
                regions.append(TokenRegion.INSTRUCTION)
                continue
            if disp.strip() == 'Answer':
                answer_seen = True
                current = TokenRegion.OTHER
                regions.append(TokenRegion.OTHER)
                continue
            if email_seen and not answer_seen:
                regions.append(TokenRegion.EMAIL)
            else:
                regions.append(current)
        return regions

    def _display_token(self, token: str) -> str:
        return token.replace('Ġ', ' ').replace('Ċ', '\n')

    def _is_control_like(self, token: str) -> bool:
        disp = self._display_token(token).strip()
        if token in _CONTROL_TOKENS:
            return True
        if 'im_start' in token or 'im_end' in token:
            return True
        if disp == '':
            return True
        return False
