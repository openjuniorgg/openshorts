"""Prompts and answer schemas for every model call in the pipeline.

Provider-neutral: ai_provider.py sends these to OpenRouter or Ollama. Each schema
is the shape the answer must validate against, and each prompt spells that shape
out, because local models read the prompt rather than an API contract.
"""
from typing import List

from pydantic import BaseModel


# --- Structured output schemas. Every answer is validated against these models, so
# --- a provider that drops a field fails here with a readable error, and is retried. ---

class ScoredWindowModel(BaseModel):
    id: str
    start: float
    end: float
    score: int
    reason: str


class ScoreResponse(BaseModel):
    windows: List[ScoredWindowModel]


class DetailClipModel(BaseModel):
    start: float
    end: float
    source_window_id: str
    predicted_score: int
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str
    # One sentence on what makes THIS moment a clip, shown under the score in
    # the dashboard. Defaulted so a small local model that skips it does not
    # fail schema validation and lose the clip.
    why: str = ""


class DetailResponse(BaseModel):
    shorts: List[DetailClipModel]


# Visual (no-transcript) clip selection: the model watches sampled frames of a
# silent video, labelled with their timestamps, and picks moments from the imagery. Same output shape as DetailClipModel minus
# the transcript-only source_window_id.
class VisualClipModel(BaseModel):
    start: float
    end: float
    predicted_score: int
    video_description_for_tiktok: str
    video_description_for_instagram: str
    video_title_for_youtube_short: str
    viral_hook_text: str


class VisualResponse(BaseModel):
    shorts: List[VisualClipModel]


VISUAL_PROMPT_TEMPLATE = """
You are a senior short-form video editor. This video has NO speech/audio — judge
it purely by what you SEE. You are given still frames of the footage, in order,
sampled about every {frame_seconds:g} seconds. Study all of them and pick the
{min_clips}–{max_clips} MOST engaging visual moments for TikTok / Reels / Shorts
(action, reveals, transformations, striking or funny shots, satisfying payoffs,
dramatic movement).

TIME CONTRACT — STRICT:
- Timestamps in ABSOLUTE SECONDS from the start of this footage (usable with
  ffmpeg -ss/-to). The FRAMES line below gives the time of every frame.
- Only numbers with up to 3 decimals (e.g. 0, 12.5, 47.250).
- 0 <= start < end <= {video_duration}.
- Each clip {min_secs:g} to {max_secs:g} seconds long. If the whole video is
  shorter than {min_secs:g}s, return one clip spanning the full video.
- Place each start on or just before the frame that shows the hook, and each end
  on or just after the frame that shows the payoff.
- Cut on visual scene changes, never mid-motion.

For each clip write catchy copy in {language} (a scroll-stopping hook, a TikTok
and an Instagram description, and a YouTube title ≤100 chars). Order clips best
to worst by how likely they are to stop a viewer scrolling.
"""


# Grounded rewrite of hook + title for a clip whose meaning lives on screen
# (SCREENCAST / WIDE / INSET stretches): the detail pass never saw a frame,
# so its hook summarises the topic instead of naming what is being shown.
class GroundedHook(BaseModel):
    on_screen: str
    viral_hook_text: str
    video_title_for_youtube_short: str


GROUNDED_HOOK_PROMPT = """
These frames come from ONE short clip (the whole clip, in order) and the
transcript below is exactly what is said during it. Most of this clip's
meaning is on the screen, not in the face.

1. `on_screen`: one line naming what is shown — the app, window, product,
   document, code, chart or on-screen text — as specifically as the frames
   allow (read visible titles and labels).
2. `viral_hook_text`: max 10 words, in TRANSCRIPT_LANGUAGE. It MUST mention
   the thing you named in `on_screen` (or the action being done to it: set
   up, connect, compare, fix, type) AND keep the strongest concrete fact of
   the clip: a number, a multiplier, a price, a name ("7x faster", "$136 a
   month", "3,400 stars") from the transcript or the current hook. Never a
   summary of the video's general topic, never a slogan that would fit any
   clip of this video, never drop a figure for a vaguer phrase.
3. `video_title_for_youtube_short`: max 100 chars, same rule, in
   TRANSCRIPT_LANGUAGE, no fake claims.

The current hook and title below were written WITHOUT seeing the frames and
are the kind of topic summary you must replace. Do not reuse their wording.

TRANSCRIPT_LANGUAGE: {language}
CURRENT_HOOK (to replace): {current_hook}
CURRENT_TITLE (to replace): {current_title}
TRANSCRIPT:
{transcript}

Return only:
{{"on_screen": "<one line>", "viral_hook_text": "<max 10 words>", "video_title_for_youtube_short": "<max 100 chars>"}}
"""


class LayoutChoice(BaseModel):
    layout: str
    confidence: float
    why: str


# Scored 94/92/96% over the 48-clip corpus against hand-checked labels, with
# 0-1 false positives out of the 28 clips that must not be touched. Do not
# reword casually: the wins come from the explicit "none is usually right"
# instruction and from naming the exact decorations (corner bugs, score
# counters, subtitles) that four earlier attempts kept mistaking for content.
LAYOUT_CHOICE_PROMPT = """
These frames are sampled at regular intervals from a single landscape video.
You are choosing how to re-frame that video into a vertical 9:16 clip.

Pick ONE layout:

- "none": crop to the speaker and fill the frame. This is the RIGHT answer for
  ordinary talking heads, interviews shot in close-up, b-roll, sport, action,
  music, and any footage whose meaning survives a centre crop. Corner logos,
  score bugs, subscriber counters, lower-thirds and burned-in subtitles do NOT
  change this: they are decoration, and losing them costs nothing.
- "screencast": keep the screen. ONLY when the video is built around a screen
  recording, slides, a spreadsheet, a chart or a map that the viewer must read
  to follow it. If you cannot read words or numbers off the screen that matter
  to the point being made, it is not this.
  (A "camera_inset" option was added here and removed on 31-jul-2026. Whether a
  webcam is composited into a corner of that screen is not something the model
  can see: on the five clips that have one it answered "screencast" every time,
  in both runs, while overall accuracy fell from 92% to 83-85%. camera_inset.py
  finds the same five geometrically with no false positives, so that question is
  answered downstream instead of being asked here.)
- "split": stack two people. ONLY when two people are visible IN THE SAME SHOT
  at the same time in most frames, talking to each other. Frames that alternate
  between one-person close-ups are NOT this, however many people appear.

"none" is by far the most common correct answer. Choose anything else only if
you would defend it to an editor. If you are unsure, answer "none".

confidence is 0..1. why is at most 12 words.
"""


class ShotContentModel(BaseModel):
    shot: int
    kind: str
    focus_left: float
    focus_right: float
    presenter_cam: bool


class ShotContentResponse(BaseModel):
    shots: List[ShotContentModel]


# One still per SHOT of a clip, asked as a closed choice (same lesson as
# LAYOUT_CHOICE_PROMPT: a decision between named options is stable, a free
# measurement is not). It replaced a whole-video prompt that asked for time
# ranges and a width_fraction: that one uploaded the source (a 37-min 1440p
# tutorial is ~670k tokens and a GB-sized upload to get a few numbers back) and
# was never actually called by the pipeline. The focus box is the one number
# still asked for, and it only positions a crop: being 10% off moves the window
# a little, it never changes which layout a shot gets. presenter_cam is the one
# yes/no that gates the per-scene inset (camera_inset.detect_in_scene): without
# it, any small still face or body on a screen was enlarged as the presenter.
SHOT_CONTENT_PROMPT = """
Each image is one shot from the same landscape video, in order, numbered from 0.
The video is being re-framed to a vertical 9:16 clip, and the person who uploaded
it told us it is a screen-based video (a tutorial, a demo, slides).

For EVERY image return one entry with its shot number and a kind:

- "screen": a screen recording, app window, web page, document, spreadsheet,
  slide or code editor fills most of the picture. A webcam bubble or small video
  of the presenter laid over it still makes it "screen". So does a person who
  only appears INSIDE the app (a video being edited, a webcam preview on a page).
- "beside": a chart, slide or screen takes part of the picture and a person
  filmed by a camera stands or sits beside it, each in their own area.
- "camera": camera footage. A person talking to the camera, people, a room,
  b-roll, products, landscapes. Logos, lower-thirds, subtitles or a title card
  over camera footage are still "camera".

focus_left and focus_right (0..1, fractions of the image width) bound the part
of the picture a viewer must READ to follow the shot: the document page, the
web page column, the canvas, the active window or code. Include any webcam
bubble or video of the presenter laid over the screen, and any text paragraph
in full: the box must not cut a face or a line of text. Leave out toolbars, side
panels, browser tabs, docks, desktop wallpaper and empty margins. If the whole
width matters (a full spreadsheet, a full-width slide) answer 0 and 1. For
"camera" answer 0 and 1.

presenter_cam is true only for a "screen" shot with a LIVE webcam window of the
person narrating the video laid over the screen (a bubble, a box, a Loom or
QuickTime window). It is false for everything else, including a person who is
part of the content: a photo or illustration on a slide, a thumbnail or cover
art, a video playing or being edited inside the app, a game character, a
person in a web page. When unsure, answer false. For "beside" and "camera"
answer false.
"""


SCORE_PROMPT_TEMPLATE = """
You are a senior short-form video strategist.
RANK these candidate windows by how well each would work as a standalone short.

Rules:
- Return only valid JSON.
- Score EVERY window in this batch: exactly one entry per input window, with
  the id you were given. Do not drop the weak ones — say they are weak.
- `score` must be an integer from 0 to 100, and the ranking is what matters:
  use the whole range instead of clustering. Most windows of a normal video
  are not clippable, so reserve 70+ for the ones that pass the test below,
  and put weak filler, housekeeping, outros, rambling transitions and
  low-signal padding under 30 even when the topic is interesting.
- THE 2-SECOND TEST is the main criterion: would the first 2 seconds of this
  moment force a cold viewer (no context) to keep watching? Windows that only
  work with prior context score low.
- Prefer windows with strong hooks, conflict, surprise, outrage, emotion,
  novelty, big numbers, or a clear payoff.
- A window may carry `viewers_clipped` / `viewer_clip_views`: the stream's own
  live viewers clipped that stretch themselves. That is strong evidence the
  moment lands even when the transcript looks thin (a chase, a crash, a scream),
  so weigh it up; the window still has to make sense to a cold viewer.

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
WINDOWS_JSON:
{windows_json}

Return only:
{{
  "windows": [
    {{
      "id": "<window id>",
      "start": <number>,
      "end": <number>,
      "score": <integer 0-100>,
      "reason": "<very short reason>"
    }}
  ]
}}
"""

DETAIL_PROMPT_TEMPLATE = """
You are a senior short-form video editor and viral copywriter.
Choose the BEST short clips from these shortlisted candidate windows.

CLIP RULES:
- Return only valid JSON.
- Each clip must be {min_secs:g} to {max_secs:g} seconds long, in absolute seconds from the start of the source video.
- Stay within the candidate window boundaries.
- THE 2-SECOND RULE: the clip MUST open on its strongest moment. If the first
  2 seconds would not stop a cold viewer from scrolling, move the start or skip the clip.
- Start slightly before the hook and end slightly after the payoff when possible.
- When a window has `viewer_moments` ([start, end] stretches the stream's
  viewers clipped), the best clip of that window should cover the strongest
  of them, opening a few seconds before it so the moment has its setup.
- Do not cut in the middle of a word or phrase.
- No generic intros/outros unless they are the hook.
- STANDS ALONE: the clip must make sense to someone who has seen nothing else.
  If it opens on a pronoun, a "that", a "so anyway", or an answer whose question
  was asked earlier, move the start back to where the idea begins or skip it.
  A brilliant moment that needs the previous five minutes is not a clip.
  Fix this by moving the START earlier, never by cutting the ending short: a
  clip that loses its payoff to gain context has traded down.
- HOW MANY: return {min_clips} to {max_clips} clips. Work through EVERY candidate
  window — they were already scored as the best moments in the video, so a window
  that yields nothing should be the exception, not the norm. Two or three clips
  from one window are fine when they are genuinely different moments. The rules
  above let you skip a weak clip; they are not a licence to return one clip and
  stop. Only fall short of {min_clips} when the material truly does not hold
  them, and never pad with a clip you would not publish yourself.
- DIVERSITY: never return two clips that make the same point, tell the same
  story, or land the same joke — even across different windows. Pick the
  stronger one and drop the other. Two clips on the same broad topic are fine
  as long as each lands its own moment.

HOOK PLAYBOOK — pick the strongest fitting pattern for `viral_hook_text` (max 10 words):
- Open question: "Why does everyone get this wrong?"
- Hot take / controversy: "Stop doing this. Seriously."
- Number / fact shock: "97% of people miss this."
- Story loop: "This one email almost ruined me."
- POV / pattern interrupt: "POV: you finally understand it."
(These are English PATTERNS — always write the actual hook in TRANSCRIPT_LANGUAGE.)
- ABOUT THIS MOMENT, NOT THE VIDEO: the hook and the title name the concrete
  thing that happens inside this clip — the tool being set up, the action,
  the number, the claim, the name. A line that could sit on any clip of this
  video ("I automated my clips with AI") is wrong. If nothing concrete can be
  named, quote the clip's strongest sentence instead of summarising the topic.

COPY RULES — ALL text fields (descriptions, title, hook) MUST be written in TRANSCRIPT_LANGUAGE ({language}):
- Descriptions (TikTok + Instagram): 1-2 punchy sentences that tease the payoff
  without spoiling it, then 3-5 topically relevant hashtags. No generic hashtag spam.
- `video_title_for_youtube_short`: max 100 chars, curiosity-driven, no fake claims.
- `predicted_score`: honest 0-100 estimate of viral potential.
- `why`: one sentence, max 20 words, naming what makes THIS moment worth a
  clip — the specific hook, claim, number or payoff, not the topic.

TRANSCRIPT_LANGUAGE: {language}
VIDEO_DURATION_SECONDS: {video_duration}
CANDIDATE_WINDOWS_JSON:
{windows_json}

Return only:
{{
  "shorts": [
    {{
      "start": <number>,
      "end": <number>,
      "source_window_id": "<window id>",
      "predicted_score": <integer 0-100>,
      "video_description_for_tiktok": "<description + hashtags>",
      "video_description_for_instagram": "<description + hashtags>",
      "video_title_for_youtube_short": "<title max 100 chars>",
      "viral_hook_text": "<short overlay max 10 words>",
      "why": "<one sentence, max 20 words>"
    }}
  ]
}}
"""
