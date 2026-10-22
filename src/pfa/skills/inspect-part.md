# Inspect one part on an assembled circuit board

You see three close-ups of the same spot on a circuit board, at the same scale and orientation:

1. **Golden**: the board built correctly. This is what the part must look like.
2. **Bare**: the same pads with solder paste, before any part was placed.
3. **As built**: the board that was just assembled. Judge this one.

Cheap image alignment has already ranked this part among the most unusual on the board, so a
defect is likely, but most parts sent to you are still fine. Report what you see.

## Verdicts

- `ok`: the same part, on its pads, the right way round. Small offsets (less than a fifth of a
  pad), slight skew, and differences in lighting, focus or solder shine are fine.
  A resistor or capacitor turned 180° is fine: they have no polarity, even though a resistor's
  printed code then reads upside down.
- `missing`: the pads are bare, as in the Bare image.
- `rotated`: the part is turned. For chips, diodes and LEDs, compare the pin-1 dot, the cathode
  band or the pin layout with the golden image. Give `rotation_deg` (90, 180 or 270, clockwise).
- `wrong_part`: a different part on the right pads: another printed code or marking, another body
  colour or size, an LED of another colour.
- `shifted`: the right part, clearly off its pads.
- `damaged_or_tombstoned`: standing on one end, cracked, or otherwise not seated.

## Rules

- Decide from what is visible. If the image does not let you tell, say `ok` and explain what you
  could not see in `observation`; an operator reviews disagreements with the measurements.
- `observation`: one sentence naming the visible evidence, for example "pin-1 dot is at the
  bottom-right, golden has it top-left".
