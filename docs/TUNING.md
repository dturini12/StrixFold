# Tuning the two boxes

What we changed on two Ryzen AI Max+ 395 boxes linked by two USB4 cables, and what each change measured. The engine's
settings named here are in `tools/dsv41-hip/serve.conf`.

## The link

- **Two cables, two jobs.** `thunderbolt0` carries the engine's control and per-token traffic, and `thunderbolt1` the
  bulk copies and the stripe. With bulk copies on the per-token cable, a reply's p95 step time rose to 41–165 ms, with
  one at 2.4 s.
- **MTU 1500, not the 65520 maximum.** On thunderbolt_net the big MTU lost packets under bulk load:

  | MTU | One-way bulk | Packets missed |
  |---:|---:|---:|
  | 65520 | 28.0 Gb/s | 4.7% |
  | 9000 | 30.9 Gb/s | 0.22% |
  | 1500 | 31.0 Gb/s | 0.04% |

  At MTU 1500 a striped 2,048-row prompt chunk went from 942 ms to 512 ms, and DwarfStar's prompt reading rose 13%
  (from 264 and 255 tok/s to 292 and 296 at 8,159 tokens). Set it on both boxes' network profiles so it survives a
  reboot.
- **Split the traffic over both cables.** A second proxy thread and a 16 KiB stripe (`COLL_PROXY_STRIPE_KIB=16`) gave
  one chat +6.6% and four chats +8.9%, with identical output.
- **Both directions share the bandwidth.** Expect about 44.5 Gb/s in total, both directions together, not twice the
  one-way number.
- **One cable costs more than its bandwidth.** On a pair of Bosgame M5s, one cable (`TP_STRIPE` empty,
  `COLL_PROXY_STRIPE_KIB=0`) lost 6–7% decode and 14–16% prompt reading against two. Each cable gets its own proxy thread,
  and one thread moves about 1.6 GB/s.
- **Link speed barely matters.** Raising the M5 pair's links from 19 to 51 Gb/s (Gen 2 to Gen 3 plus AMD's host-to-host
  setting) gave +1–2% decode and +6% prompt reading. Decode is limited by latency on small exchanges, not bandwidth.
  Details: [BOSGAME-M5.md](BOSGAME-M5.md#the-usb4-links).

## TCP retransmits

The engine's small per-token messages need a short retransmit floor, so a lost packet doesn't stall a reply. With the
same short floor (`TCP_RTO_MIN_US=8000`) on a socket carrying 42 MB prompt exchanges, three of five 8K prompt reads
stalled: the receiver ended 3–7 MB short and hit its deadline. The fix is a short floor for the small messages
(`COLL_RTO_MIN_US=8000`) and a 200 ms floor during bulk phases.

## Power

The boxes serve at the default balanced power setting. The CPU's performance setting is faster: DwarfStar went from
13.3 to 14.7 tok/s. It costs more power, so it is off by default here.

- **Cool a Bosgame M5 before you give it more power.** Its stock fan curve lets the GPU sit at 84–88 °C at 85 W. With
  the fans driven harder (the EC's duty override), the same load gave 4% more output at 64–70 °C. Its performance mode
  with the stock fans was 9% slower, at 95–100 °C; with the faster fans a GPU still held 95 °C for 30 s within 91 s.
  Details: [BOSGAME-M5.md](BOSGAME-M5.md#cooling).

## Several chats

With drafting on, identical prompts decode in lockstep, and leftover drafts break that. Turning drafting off from six
chats at once (`DRAFT_BYPASS_AT=6`) raised eight chats from 97.8 to 108 tok/s, with the output unchanged.

A long new prompt beside chats that are answering is read in slices (`FILL_SLICE_MS`, 1,000 by default), with a round
for the others between slices. Beside three answering chats, a 100K prompt left them waiting at most 2 s at a time,
against 71 s when read whole, and itself took 76 s instead of 72. A shorter slice gives the others more rounds, down to
one chunk a slice (where 500 and 1,000 already are); 2,000 reads a little faster and makes them wait up to 3 s. 0 reads
every prompt whole.

A request that arrives while such a prompt is being read waits for the whole read, minutes for a very long one.
`FILL_CUT_IN=4096` lets a waiting request with at most that many tokens to read, without images, go first: between two
slices its prompt is read whole and it starts answering, then the long read goes on. The tokens to read are the prompt
past the longest state the prompt cache resumes for it, so an agent's next turn on a long kept history counts only its
new tail; a turn whose state went stale, or is lost when its pass is placed, waits for the long read instead. One
request cuts in per slice, so the long read never stops, and it needs a free stream slot and room in the pool like any
request. The others wait for the short prompt's read on top of the slice, so keep the size small. With it set, a long
prompt is read in slices even when nothing else is answering, so that a request can cut in. It is off by default and
needs `FILL_SLICE_MS` above 0.

## Prompt reading and the drive

Prompt reading also reads each token's Engram rows from the tables on the drive, two 4 KiB page reads a row unless the
page cache has them, so a drive whose random reads slow down, or a box with little page cache left, makes it swing.
`ENGRAM_ROW_CACHE_MIB=512` keeps rows in memory dense (264 bytes a row against a page), so a text read again (a second
question on a long document, a turn whose kept state went) skips the drive: on a pair with DRAM-less drives a 25K
re-read took 14.0 s with it against 23.6 s without, the same logits either way. It costs that much host memory on each
rank and does little for a text read for the first time. Empty or 0: off.

## Pictures

Five settings for requests with images, all off by default; each leaves the output unchanged.

- `FILL_IMAGES=1` reads a prompt with pictures in `FILL_SLICE_MS` slices like a text prompt, so the chats that are
  answering keep getting rounds while it is read, and a short request (`FILL_CUT_IN`) may cut in on it. Off, such a
  prompt is read whole and the others wait for it. It needs `FILL_SLICE_MS` above 0.
- `TOWER_SLICE_MS=200` lets the vision tower run only between the answering chats' rounds, about that long each
  time, instead of beside them: a round then waits for at most one tower block, not a whole picture (3-4 s). A
  picture takes a little longer to encode while others answer; with nothing answering the tower runs free.
- `IMAGE_ROWS_CACHE_MIB=64` keeps the tower's output for recent pictures in rank 0's RAM (about 10.5 MiB a
  full-size picture), so a turn that repeats a picture does not run the tower for it again.
- `IMAGE_ROWS_BUDGET_MIB=1280` bounds the tower output that rank 0's encoder holds for requests: an encode that would
  pass it waits, holding no memory, until earlier requests' rows are freed (one request alone, or one a prompt pass
  waits for, always runs). It bounds host memory when several screenshot-heavy requests arrive at once.
- `IMAGE_MEMO=65536` makes the server remember each picture's preparation by its bytes, and prepare a picture's
  payload only when the tower encodes it, which saves time and host memory on long screenshot histories.
