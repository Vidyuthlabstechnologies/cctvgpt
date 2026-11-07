# Where CCTV decides the case: Indian use-case catalogue

This catalogue lists the situations in which CCTV footage is decisive in Indian investigations, prosecutions, claims and operations. For each one it states what the investigator needs from the footage and what CCTVGPT does today. Status is as of November 2026:

- **Built**: available and measured.
- **Partial**: available but weak, or not measured on that task.
- **Gap**: not built.

Statistics and case references marked **[VERIFY]** are from memory and must be checked against the primary source (NCRB *Crime in India*, MoRTH *Road Accidents in India*, court judgments) before external use.

## Why footage decides so many Indian cases

- **Volume:** Indian cities have some of the highest densities of CCTV cameras in the world. Hyderabad and Delhi are often cited **[VERIFY current counts]**. Most cameras are privately owned (shops, homes, apartments), so investigators collect footage from dozens of owners after an incident.
- **Witnesses turn hostile:** many trials depend on independent evidence, and footage is often the only neutral witness.
- **Admissibility:** electronic records are admissible under Section 63 of the Bharatiya Sakshya Adhiniyam (BSA), 2023, which replaced Section 65B of the Evidence Act. A certificate is needed, and courts expect an unbroken chain of custody. That is why CCTVGPT exports are cut from the original file with hashes and an audit trail.
- **The bottleneck is review, not cameras:** footage is overwritten in 7–30 days on typical DVRs **[VERIFY]**, so it must be collected and searched fast.

## Crimes against the person

| Case type | What the investigator needs from footage | CCTVGPT today | Status |
|---|---|---|---|
| Chain snatching, mobile snatching (typically two people on a two-wheeler) | Two-wheeler, riders' clothing and helmet, plate, escape route across cameras | Vehicle and description search; plate search; cross-camera trail | Partial: two-wheeler plates are small, and helmets hide faces |
| Murder, assault | Who was present, when they arrived and left, weapon, the fight itself | Description search; violence detection (see `docs/SEARCH.md`); timeline across cameras | Partial |
| Kidnapping, missing children and missing elderly | Last-seen location, who they left with, vehicle | Description search ("girl in school uniform"); face search with case justification; trail | Partial: face accuracy at night is limited |
| Crimes against women: stalking, harassment, acid attack | Repeated presence of the same person or vehicle near the victim; the attack itself | Plate and face recurrence across days; description search | Partial |
| Mob violence and riots | Sequence of events, who started it, weapons | Violence detection, description search | Partial: identification of individuals in crowds must be case-scoped and audited (high misuse risk) |
| Custodial violence | Footage inside police stations. The Supreme Court ordered CCTV in all police stations (*Paramvir Singh Saini v. Baljit Singh*, 2020) **[VERIFY]** | Violence detection on station footage; tamper-evident audit | Partial: an oversight use as well as an investigative one |
| Violence against doctors in hospitals | Attackers, sequence | Violence and description search | Partial |

## Property crime

| Case type | What the investigator needs | CCTVGPT today | Status |
|---|---|---|---|
| Vehicle theft, the most common property crime in many cities **[VERIFY NCRB]** | Stolen vehicle's plate across junction cameras; thief's face or clothing | Plate search with fuzzy matching; trail across cameras with "where to look next" | Built (plate accuracy limits apply) |
| House-breaking and burglary (often at night) | Entry time, number of people, vehicle used, route | Night enhancement; description and vehicle search | Partial: night footage is the hard case |
| Jewellery shop and bank robbery, ATM theft | Faces, vehicles, entry and exit routes, weapons | Face (opt-in), plate, trail; weapon detection | Gap: no weapon detector |
| ATM skimming, and cash withdrawals by cyber-fraud mules | Person at the ATM at a transaction time supplied by the bank | Time-indexed search; face recurrence across ATMs | Partial: needs bank timestamp import |
| Retail, warehouse and logistics theft (private sector) | Who removed goods, when; loading vehicles | Description, vehicle and plate search | Partial |

## Road and transport

| Case type | What the investigator or insurer needs | CCTVGPT today | Status |
|---|---|---|---|
| Hit-and-run. India has about 1.7 lakh road deaths a year **[VERIFY MoRTH]** | Fleeing vehicle type, colour, partial plate, route | Accident detection (see `docs/SEARCH.md`); vehicle colour and type search; partial plate search; trail | Partial |
| Motor accident claims (MACT) and insurance fraud | Who was at fault, speed, signal state, whether the accident was staged | Accident event search; clip export with chain of custody | Partial: no speed or signal-state estimation yet |
| Drunk, rash or wrong-way driving | Vehicle behaviour over time | Gap: no trajectory analytics | Gap |
| Railway station crime and accidents | Platform events, falls from trains, theft | Fall/lying-person search, description search | Partial |
| Toll plaza and checkpost movement | Vehicle passage times | Plate trail across cameras with clock correction | Built |

## Public order and safety

| Case type | What is needed | CCTVGPT today | Status |
|---|---|---|---|
| Crowd crushes and stampedes (for example religious gatherings and stadium events) **[VERIFY specific incidents]** | Crowd density rising over time; the moment flow stopped | Gap: no crowd density estimation | Gap |
| Fires (commercial buildings, hospitals, game zones) | Ignition point and time, exits blocked, response time | Gap: no fire or smoke detector | Gap |
| Terror and blast investigations | Suspect's movement across many cameras and buses over hours; bag drop | Description and face search; cross-camera trail; time-synced map | Partial: needs person re-identification across cameras |
| Election strongrooms, exam halls | Proof that nobody entered | Activity search over long, static footage | Partial |

## Other

| Case type | Needs | CCTVGPT today | Status |
|---|---|---|---|
| Workplace accidents (factories, construction) | Fall, machine incident, PPE worn | Fall search; PPE detection | Gap (PPE) |
| Animal cruelty, cattle smuggling at checkposts | Vehicle and plate at checkposts | Plate trail | Built |
| Illegal dumping, encroachment | Vehicle and time | Vehicle and plate search | Partial |
| Civil disputes (property damage, neighbour disputes) | Who did what, when | Description search, clip export | Partial |

## The gaps that matter most, in priority order

1. **Person re-identification across cameras** (clothing and gait, not face). It answers "where did this person go?", which matters in most of the cases above. It is also less invasive than face recognition.
2. **Weapon detection** (knives, sticks, firearms) for robbery and assault.
3. **Two-wheeler plates and riders**: most snatchings use two-wheelers, and their plates are small and often two-line.
4. **Crowd density, fire and smoke**: for public-safety operations rather than investigations.
5. **Speed and trajectory from fixed cameras**: for accident fault and rash driving.
6. **Indian evaluation data**: every accuracy figure in `docs/SEARCH.md` comes from public, mostly non-Indian datasets. A pilot with a police unit (see the pilot proposal) is the only way to get honest numbers on Indian footage.
