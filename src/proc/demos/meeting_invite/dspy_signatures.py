import dspy


class EmailToMeetingInfo(dspy.Signature):
    """
    You are a scheduling assistant.  Your job is to read a meeting-proposal email
    and extract structured scheduling data from it.

    Rules:
    - duration_minutes: if not stated, default to 30.
    - sender_iana_timezone: infer from city/region clues (e.g. "San Francisco" ->
      "America/Los_Angeles").  If genuinely unknown, return "UNKNOWN".
    - urgency: pick the most specific one: "today" > "this_week" > "next_week" > "flexible".
    - flexibility: "specific" if the sender named concrete times; "flexible" if open-ended.
    - preferred_windows: list every time-window the sender explicitly proposed.
      Each window may have day_of_week, date, time_of_day (HH:MM or "morning"/
      "afternoon"/"evening"), iana_timezone, and duration_hint.
    - meeting_topic: short phrase for the meeting purpose, or null.

    Output ONLY valid JSON matching the schema below - no prose, no markdown fences.

    JSON schema:
    {
      "sender_iana_timezone": "<IANA tz string or UNKNOWN>",
      "duration_minutes": <int>,
      "urgency": "<today|this_week|next_week|flexible>",
      "flexibility": "<specific|flexible>",
      "preferred_windows": [
        {
          "day_of_week":   "<Monday...Sunday or null>",
          "date":          "<YYYY-MM-DD or null>",
          "time_of_day":   "<HH:MM or morning/afternoon/evening or null>",
          "iana_timezone": "<IANA tz string or null>",
          "duration_hint": <int or null>
        }
      ],
      "meeting_topic": "<string or null>"
    }
    """

    email_from: str = dspy.InputField(desc="Sender's email address")
    email_to: str = dspy.InputField(desc="Recipient's email address")
    email_body: str = dspy.InputField(desc="Full email body text")
    current_date: str = dspy.InputField(desc="Today's date as YYYY-MM-DD (for relative references)")

    extracted_json: str = dspy.OutputField(
        desc="JSON object matching the schema described in the docstring",
    )
