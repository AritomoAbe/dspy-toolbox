from enum import Enum

from pydantic import BaseModel

_EPSILON: float = 1e-9
_TOP5: int = 5


class PromptSegmentType(Enum):
    """
    The logical regions of a compiled DSPy prompt.
    Each region plays a different role in driving the model's output.
    """
    INSTRUCTION = "instruction"   # The task description / signature instructions
    DEMO = "demo"                 # A single bootstrapped few-shot example
    INPUT = "input"               # The actual query being scored


class SegmentAttribution(BaseModel):
    """
    Attribution score for a single prompt segment on a single example.

    attribution_score:  score_full - score_ablated.
                        Positive  → segment contributes to correct output.
                        Near zero → segment is being ignored.
                        Negative  → segment is hurting the model (adversarial).
    segment_type:       INSTRUCTION, DEMO, or INPUT.
    segment_index:      For DEMO segments, the index of the demo (0-based).
                        For INSTRUCTION and INPUT, always 0.
    segment_preview:    First 80 chars of the segment text, for readability.
    score_full:         Score when no segment is masked.
    score_ablated:      Score when this segment is replaced with a blank.
    """
    segment_type: PromptSegmentType
    segment_index: int
    segment_preview: str
    attribution_score: float
    score_full: float
    score_ablated: float


class ExampleAttribution(BaseModel):
    """Attribution breakdown for a single dataset example."""
    example_index: int
    score_full: float
    segments: list[SegmentAttribution]

    @property
    def instruction_attribution(self) -> float:
        scores = [s.attribution_score for s in self.segments
                  if s.segment_type == PromptSegmentType.INSTRUCTION]
        return scores[0] if scores else float()

    @property
    def demo_attributions(self) -> list[float]:
        return [s.attribution_score for s in self.segments
                if s.segment_type == PromptSegmentType.DEMO]

    @property
    def avg_demo_attribution(self) -> float:
        scores = self.demo_attributions
        return sum(scores) / len(scores) if scores else float()


class AttributionResult(BaseModel):
    """
    Aggregate attribution results across the full dataset.

    avg_instruction_attribution:  Mean attribution score of the INSTRUCTION segment.
                                  Low value → instruction text is being ignored.
    avg_demo_attribution:         Mean attribution score across all DEMO segments.
                                  Low value → few-shot examples are not doing real work.
    per_demo_avg_attribution:     Per-demo-index mean attribution, revealing which
                                  specific demos are contributing vs. dead weight.
    example_attributions:         Full per-example breakdown for drill-down.
    n_examples:                   Total examples evaluated.
    n_demos:                      Number of few-shot demos in the compiled prompt.

    Interpretation guide:
        instruction_attribution > 0.1  → instruction is actively used
        instruction_attribution < 0.05 → model is ignoring the instructions
        avg_demo_attribution > 0.1     → few-shot examples are doing real work
        avg_demo_attribution < 0.02    → demos are dead weight; consider removing them
        per_demo_avg shows large variance → some demos are harmful, prune them
    """
    avg_instruction_attribution: float
    avg_demo_attribution: float
    per_demo_avg_attribution: dict[int, float]   # demo_index -> mean attribution
    example_attributions: list[ExampleAttribution]
    n_examples: int
    n_demos: int


# ── Token-level saliency (Panel A) ────────────────────────────────────────────

class TokenSaliency(BaseModel):
    """
    Normalized attribution score for one prompt token.

    token:             Decoded token string.
    index:             Position in the flat prompt token sequence.
    saliency:          Normalized [0, 1] attribution score.
                       L2-norm of embedding-gradient vector, normalized
                       by the max across all tokens in this example.
    raw_norm:          Un-normalized L2 norm (for aggregation).
    segment_label:     Which DSPy segment this token belongs to:
                       'instruction', 'demo_N', or 'input'.
    """
    token: str
    index: int
    saliency: float
    raw_norm: float
    segment_label: str


# ── Logit lens per layer (Panel B) ───────────────────────────────────────────

class LayerProbe(BaseModel):
    """
    Logit-lens result for one decoder layer.

    layer_index:      0-based layer number.
    target_prob:      Probability of the target token at this layer.
    target_rank:      Rank of target token at this layer (1 = top-1).
    first_top1:       True if this is the first layer where rank == 1.
    """
    layer_index: int
    target_prob: float
    target_rank: int
    first_top1: bool = False


# ── Segment attribution (Panel C) ────────────────────────────────────────────

class SegmentSaliency(BaseModel):
    """
    Average saliency for a logical DSPy prompt segment.

    label:          'instruction', 'demo_N', or 'input'.
    text_preview:   First 60 chars of the segment text.
    avg_saliency:   Mean normalized saliency across all tokens in segment.
    token_count:    Number of tokens in this segment.
    """
    label: str
    text_preview: str
    avg_saliency: float
    token_count: int


# ── Per-example result ────────────────────────────────────────────────────────

class LIGExampleResult(BaseModel):
    """
    Full LIG attribution result for one dataset example.

    example_index:      0-based index.
    target_text:        The prediction text used as Captum's attribution target.
    target_token:       First token of target_text (what LIG attributes toward).
    target_prob:        Final-layer probability of target_token.
    target_rank:        Final-layer rank of target_token.
    convergence_delta:  Captum convergence delta — should be < 0.05 for trust.
    top1_reached_layer: Layer index where target first became rank-1.
                        -1 if it never reached rank-1.
    token_saliencies:   Per-token attribution scores (Panel A).
    layer_probes:       Per-layer logit-lens results (Panel B).
    segment_saliencies: Per-segment averaged saliency (Panel C).
    """
    example_index: int
    target_text: str
    target_token: str
    target_prob: float
    target_rank: int
    convergence_delta: float
    top1_reached_layer: int
    token_saliencies: list[TokenSaliency]
    layer_probes: list[LayerProbe]
    segment_saliencies: list[SegmentSaliency]

    @property
    def top5_tokens(self) -> list[TokenSaliency]:
        ranked = sorted(self.token_saliencies, key=lambda t: t.saliency, reverse=True)
        return ranked[:_TOP5]

    @property
    def input_vs_boilerplate_ratio(self) -> float:
        """
        Ratio of input-segment saliency to instruction-segment saliency.
        > 1.0 → model attends to the actual input more than boilerplate (good).
        < 1.0 → model is latching onto prompt structure (concerning).
        """
        input_seg = next((s for s in self.segment_saliencies if s.label == "input"), None)
        instr_seg = next((s for s in self.segment_saliencies if s.label == "instruction"), None)
        if not input_seg or not instr_seg or instr_seg.avg_saliency < _EPSILON:
            return float()
        return input_seg.avg_saliency / instr_seg.avg_saliency


# ── Aggregate result across dataset ──────────────────────────────────────────

class LIGAttributionResult(BaseModel):
    """
    Aggregate Layer Integrated Gradients attribution across the full dataset.

    Panels map to the analysis from the attached reference implementation:
        Panel A → token saliency (which tokens drive the prediction)
        Panel B → logit lens (which layer the model 'decides')
        Panel C → segment saliency (instruction vs demos vs input)

    avg_input_saliency:       Mean saliency of the 'input' segment.
                              Should be the highest — model routes signal from input.
    avg_instruction_saliency: Mean saliency of the 'instruction' segment.
                              Low → instruction is boilerplate, not read.
    avg_demo_saliency:        Mean saliency of demo segments.
                              Low → few-shot examples are ignored.
    avg_convergence_delta:    Mean Captum convergence delta across examples.
                              Values > 0.05 mean attribution estimates are unreliable.
    pct_examples_top1_reached: Fraction of examples where target reached rank-1
                               in at least one layer. Low → prompt cannot elicit answer.
    avg_top1_layer:           Mean layer index where target first becomes rank-1.
                              Earlier = model decides sooner = more confident.
    top_saliency_tokens:      Most salient tokens globally (aggregated across examples).
    model_name:               HuggingFace proxy model used.
    n_examples:               Total examples evaluated.
    example_results:          Full per-example breakdowns.

    Interpretation guide (matches Panel C logic from reference):
        avg_input_saliency > avg_instruction_saliency  → good
        avg_demo_saliency  < 0.05                      → demos are dead weight, prune
        pct_examples_top1_reached < 0.5                → add demos or rephrase signature
        avg_convergence_delta > 0.05                   → increase ig_steps for reliability
    """
    avg_input_saliency: float
    avg_instruction_saliency: float
    avg_demo_saliency: float
    avg_convergence_delta: float
    pct_examples_top1_reached: float
    avg_top1_layer: float
    top_saliency_tokens: list[TokenSaliency]
    model_name: str
    n_examples: int
    example_results: list[LIGExampleResult]
