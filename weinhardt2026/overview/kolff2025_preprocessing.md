# kolff2025 — preprocessing of the grooming-negotiation data

Implemented in `weinhardt2026/studies/kolff2025/preprocessing_kolff2025.py` (SPICE repo). Run from the repo root:
`python weinhardt2026/studies/kolff2025/preprocessing_kolff2025.py`.

This file records how the raw chimpanzee grooming data are recoded into the
behavioural categories used for modelling, and why. The categories were agreed
on between Kayla Kolff, Hyerim Hwang, Sebastian Musslick and Daniel Weinhardt
(email thread, 20–31 Aug 2026).

## Source file

`Original_data_with_dominance_rank - with dominance rank_ turntaking_df copy.csv`
holds the element-level coding: `SigAct_ID1/2` as behaviour names (e.g.
`present_back`, `push`), ape name codes, dominance ranks, `interaction_id` and
`community_id` (CE/WE). It has 6334 rows, 311 interactions and 41 apes.

An empty cell (`NaN`) in `SigAct_IDx` means that ape did nothing coded in that
event. In every row at least one ape acts.

The older `kolff2025_original.csv` codes the same events into only 5 coarse
codes and is not used. In that file, `reposition` and `maintain contact` were
part of the "action" code.

## Terminology

`Behavioural_Group > Behavioural_Category > Behavioural_Element`

- **Element**: the original coded behaviour name (`SigAct_ID*` value).
- **Category**: the modelling unit (table below).
- **Group**: *Groom* (the goal/outcome behaviour) vs *Negotiation* (all
  behaviours that may lead to grooming). The name "Negotiation" replaces
  "Signal" at Kayla's request, because in primatology "signal" is reserved for
  gestures, vocalisations and facial expressions.

## Category mapping

| Group | Category | Elements | TOBETO | n (ID1 + ID2) |
|---|---|---|---|---|
| Groom | Groom | `groom` | — (outcome) | 2583 |
| Negotiation | Self_Reposition | `reposition` | expect **to be** groomed | 814 |
| Negotiation | Reposition_Body | `touch`, `grab-pull limb`, `push`, `touch hold` | **to** groom | 707 |
| Negotiation | Grooming_Process | `maintain contact` | **to** groom | 657 |
| Negotiation | Grooming_Solicitation | `present_*` (arm, back, body, face, genitals, head, limb, rump, torso), `raise *` (arm, hand, head, leg), `extend *` (arm, leg), `kiss`, `hold` | expect **to be** groomed | 582 |
| Negotiation | Directed_Scratch | `directed scratch` | expect **to be** groomed | 567 |

What the categories mean:

- **Self_Reposition**: the ape changes its own posture without clearly
  presenting a new body part (otherwise it is coded `present`). This was
  previously part of the coarse "action" code.
- **Reposition_Body**: a gesture by which one ape (typically the groomer)
  moves the partner into a position for further grooming. It is distinct from
  Self_Reposition.
- **Grooming_Process**: `maintain contact`. Grooming pauses shorter than 2 s
  are still coded as grooming. If the groomer's hand stays on the partner
  after grooming, the behaviour is coded `maintain contact`. Grooming usually
  resumes afterwards (groom → maintain contact → groom). This was previously
  part of the coarse "action" code.
- **Grooming_Solicitation**: visual and tactile requests to be groomed, merged
  because the tactile elements are rare. `hold` is ambiguous in the
  literature: Kayla lists it as "Grooming solicitation (request to be
  groomed), reposition body (move body into indicated position)". It is
  assigned here following Kayla's later note that it is one of the tactile
  gestures for requesting to be groomed, and Hyerim's merge of those gestures
  into Solicitation.
- **Directed_Scratch**: kept separate from Solicitation. It can also express
  social anxiety, or occur with no partner nearby.
- `kiss` (1 event) is too rare to matter and stays in Solicitation as
  classified.

## Excluded elements

Any row containing one of these elements is **deleted entirely**; it is not
recoded as "no act".

| Elements | Reason |
|---|---|
| `approach`, `follow`, `mount` | Socially directed actions that are goals in themselves, not part of negotiating grooming (Kayla, Sebastian). `approach` opens 41 bouts and `follow` closes 15, so deleting them mostly trims the edges of bouts. |
| `leave`, `move away` | Exits from the interaction, not negotiation. 199 of 311 bouts end with one of them, so deleting them mostly trims bout ends. `leave` is nearly always terminal (161/212 last event; mid-bout it is followed almost only by the partner's reaction). `move away` is sometimes only a pause: in 13 of its 20 mid-bout cases grooming resumes. It is excluded anyway, for simplicity. |
| `handclasp` | A span around mutual grooming, not a separate act (see below). Rare (22 rows). |
| `peer`, `leaf groom` | Function unclear (Kayla's classification). Too few (41) to form a category, and an "unclear" category would not be interpretable. |
| `display`, `drumming` | Not covered by the categorisation; 2 events each |

**Handclasp.** Every handclasp is coded as a long span (e.g. 138.5–170.8 s). It
covers both apes grooming each other, and each ape's grooming is *already*
coded as a separate `groom` row inside that span, in all 20 handclasp
episodes. Deleting the handclasp rows therefore loses no grooming. Splitting
them into two groom events would count the grooming twice. The order of the two
grooms is given by those existing rows.

## Resulting sizes

`kolff2025_categories.csv`: 5905 rows (429 deleted), 311 interactions. All
interactions keep at least one row, and the median bout length drops from 15
to 13 events. 12 bouts are left with fewer than 2 rows. They contain no
transition to predict and are dropped when building sequences.

## Simultaneous acts

Five remaining rows have both apes acting at once (`groom`+`groom` ×2,
`present_arm`+`directed scratch`, `push`+`raise head`, `push`+`raise leg`).
Both acts are kept. When the outcome is ambiguous (both groom), it is resolved
focal-first: from each ape's own perspective, the event counts as that ape
grooming. The two `groom`+`groom` rows are mutual grooming at a handclasp onset.

## Event order

**Each interaction is one sequence (block), and its row order is the event
order.** The model is event-based: one row is one time step. Timestamps are not
used. This is why:

- **Row order is already chronological.** The older file's timestamps confirm
  it: they follow the row order apart from a few overlapping events.
- **Timestamps restart within an interaction.** Many interactions span several
  video files of about 5 minutes each (110 of 311). `start` restarts at 0 in
  each file, so it is not a usable sort key. The current
  `benchmarking_kolff2025_groom.build_perspective_dataframe` sorts by `start`
  and thereby scrambles those interactions. It must use row order instead.
- **A fixed time grid would be dominated by grooming.** Grooming takes 89% of
  observed time but only 41% of events (median 25 s, versus 1–3 s for
  negotiation behaviours). A 2 s grid would turn about 6300 events into about
  54k steps, mostly continued grooming, while several short negotiation
  behaviours would fall into one bin.

## Output columns

| Column | Meaning |
|---|---|
| `interaction_id` | as in the source (1-based); one sequence per interaction |
| `community_id` | CE / WE |
| `ID1`, `ID2` | ape name codes |
| `rank_ID1`, `rank_ID2` | raw dominance ranks (normalised within community downstream) |
| `SigAct_ID1`, `SigAct_ID2` | original element; NaN = no act |
| `Category_ID1`, `Category_ID2` | category from the table above; NaN = no act |
