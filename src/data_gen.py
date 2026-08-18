"""
data_gen.py
-----------
Generates ~800 labelled municipal complaint samples across 6 departments.
Uses template-based generation with Odisha place names, temporal markers,
and urgency phrases to produce realistic Indian civic text.

Run:
    python src/data_gen.py
Output:
    data/raw/complaints_raw.csv
"""

import random
import csv
import os

random.seed(42)

# ── Place names ──────────────────────────────────────────────────────────────
PLACES = [
    "Rasulgarh", "Unit-4", "Unit-6", "Saheed Nagar", "Nayapalli",
    "Patia", "Chandrasekharpur", "Khandagiri", "Bhubaneswar Old Town",
    "Cuttack", "Puri", "Sambalpur", "Brahmapur", "Rourkela",
    "Master Canteen Square", "Vani Vihar", "Jaydev Vihar", "IRC Village",
    "Acharya Vihar", "Barmunda", "Sishu Bhawan Chowk", "Bomikhal",
    "Tamando", "Aiginia", "Dumduma", "Kalinga Nagar",
]

STREET_TYPES = ["road", "lane", "marg", "street", "nagar", "colony", "chowk", "square"]

TEMPORAL = [
    "for three days", "since last week", "for over a week",
    "for two weeks now", "since Monday", "for the past four days",
    "since last month", "for more than a week", "repeatedly",
    "again and again", "still not fixed", "even after multiple complaints",
]

URGENCY = [
    "it is causing inconvenience to residents",
    "children are at risk",
    "it is a public safety hazard",
    "elderly people are affected",
    "please take immediate action",
    "urgent attention required",
    "the situation is getting worse daily",
    "daily commuters are suffering",
    "the whole locality is inconvenienced",
    "we have been suffering daily",
    "this needs to be fixed immediately",
    "motorists are in danger",
    "there could be a serious accident",
]

def rplace():
    return random.choice(PLACES)

def rtemp():
    return random.choice(TEMPORAL)

def rurgency():
    return random.choice(URGENCY)

# ── Template banks per department ────────────────────────────────────────────

ROADS_TEMPLATES = [
    "The road near {place} has a large pothole {temp}. {urgency}.",
    "There is a huge crater on the main road near {place} {temp}. {urgency}.",
    "The {street} near {place} is completely broken with multiple potholes. {urgency}.",
    "Road near {place} has been heavily damaged due to recent digging work {temp}.",
    "The footpath on {place} {street} is broken and dangerous for pedestrians.",
    "Construction debris has been dumped on the road near {place} {temp}. {urgency}.",
    "The road surface near {place} has completely eroded. Vehicles are getting damaged.",
    "Heavy potholes on {place} {street} are causing accidents {temp}.",
    "The road divider near {place} is broken and there is no barricading. {urgency}.",
    "Road cutting work was done near {place} but it has not been repaired {temp}.",
    "Broken road near {place} is flooded with water during rains. {urgency}.",
    "The service road near {place} is completely damaged with loose gravel everywhere.",
    "Potholes on the road to {place} have still not been filled {temp}. {urgency}.",
    "The bridge near {place} has damaged railings. It is very unsafe.",
    "Road maintenance work near {place} was started but left incomplete {temp}.",
    "The road in {place} colony has completely broken down after the monsoon. {urgency}.",
    "Large potholes near {place} are causing traffic diversions {temp}.",
    "The road from {place} to the main highway is in terrible condition.",
    "Footpath near {place} school has broken slabs which are dangerous for children.",
    "Road near {place} has been dug for cable laying and not repaired {temp}. {urgency}.",
    "The road near {place} is full of potholes making two wheelers prone to accidents.",
    "Concrete road near {place} has developed cracks and is sinking in some areas.",
    "The road near {place} is completely dark at night and full of potholes. {urgency}.",
    "Road dividers on {place} {street} are broken and posing danger to motorists.",
    "Speed breaker near {place} has no markings and is causing accidents.",
    "The main arterial road near {place} has not been repaired {temp}. {urgency}.",
    "Broken road tiles near {place} are creating hazard for pedestrians {temp}.",
    "Road near {place} gets flooded every time it rains due to poor surface drainage.",
]

WATER_TEMPLATES = [
    "There has been no water supply in {place} {temp}. {urgency}.",
    "The water pipe near {place} has burst and water is being wasted {temp}.",
    "Dirty and foul-smelling water is coming from taps in {place} {temp}. {urgency}.",
    "Water pressure is very low in {place} {street}. We are not getting enough water.",
    "The water supply in {place} has been disrupted {temp}. {urgency}.",
    "A water pipeline near {place} is leaking heavily, causing road damage too.",
    "No water supply in our area near {place} since the maintenance work started {temp}.",
    "Water coming from the taps in {place} is brown and unfit for drinking. {urgency}.",
    "The underground water pipeline near {place} is broken and water is gushing out.",
    "We have not received water supply in {place} {temp}. {urgency}.",
    "Water tanker has not come to {place} {temp}. Residents are suffering.",
    "There is a leaking water main near {place} {street}. Please fix urgently.",
    "The borwell near {place} is not working {temp}. No drinking water available.",
    "Water pipe near {place} burst during road construction and has not been repaired.",
    "Supply water in {place} has high chlorine smell. Not suitable for cooking. {urgency}.",
    "The overhead water tank in {place} is leaking and overflowing on the road.",
    "No drinking water in {place} colony {temp}. Residents are buying water at high cost.",
    "Water pipeline near {place} has developed multiple leaks causing waterlogging.",
    "Community water tap near {place} has been broken {temp}. {urgency}.",
    "The water meter in our building near {place} is faulty and showing excess usage.",
    "Sewage is mixing with drinking water supply in {place}. Very serious. {urgency}.",
    "Water supply timings in {place} are erratic. We never know when water will come.",
    "The main water supply valve near {place} is stuck and causing low pressure issues.",
    "Water supply has been cut near {place} without any prior notice {temp}. {urgency}.",
    "Old water pipes in {place} are rusting and contaminating drinking water supply.",
]

ELECTRICITY_TEMPLATES = [
    "The streetlight near {place} has been not working {temp}. {urgency}.",
    "There is frequent power cut in {place} {temp}. {urgency}.",
    "The electric pole near {place} has a loose wire hanging dangerously.",
    "Power supply in {place} goes off every night {temp}. We face a lot of issues.",
    "The transformer near {place} has been making loud noise and sparking {temp}.",
    "Streetlights on {place} {street} have been out {temp}. Area is unsafe at night.",
    "There is a live electric wire lying on the road near {place}. Very dangerous. {urgency}.",
    "Power outage in {place} {temp}. All perishable food is getting wasted.",
    "The electric meter box near {place} is open and exposed to rain. Safety risk.",
    "Frequent voltage fluctuations in {place} are damaging our appliances {temp}.",
    "The electricity pole near {place} has tilted and could fall any moment. {urgency}.",
    "Power supply restored after 8 hours but now going off again near {place}. {urgency}.",
    "Street lights in {place} are on during daytime and off at night {temp}.",
    "The electric connection to {place} community hall has been disconnected wrongly.",
    "Broken street light near {place} school is causing safety concern for students.",
    "Electricity transformer near {place} has broken down {temp}. Whole street affected.",
    "Overhead electric cable near {place} is sagging very low. Vehicles might touch it.",
    "Power cuts in {place} are lasting more than 6 hours daily {temp}. {urgency}.",
    "The electricity distribution box near {place} is exposed and unlocked. Danger risk.",
    "Our electricity bill near {place} has come 3 times higher than normal without reason.",
    "Electric pole in {place} colony is leaning and might fall during rain. {urgency}.",
    "Streetlights installed near {place} are not turning on since installation {temp}.",
    "High tension wire near {place} is passing very low over the road. Very dangerous.",
    "New electricity connection applied for near {place} but not provided {temp}.",
    "Power supply fluctuates heavily near {place} causing damage to electronic devices.",
]

SANITATION_TEMPLATES = [
    "Garbage has not been collected in {place} {temp}. {urgency}.",
    "The drain near {place} is overflowing with sewage {temp}. {urgency}.",
    "Open garbage dumping is happening near {place} {street}. Foul smell everywhere.",
    "The garbage van has not visited {place} {temp}. Waste is piling up on the road.",
    "Sewage drain near {place} is blocked and water is overflowing on the road. {urgency}.",
    "There is a large pile of garbage near {place} that has not been cleared {temp}.",
    "Open drainage near {place} is breeding mosquitoes. Residents are falling sick.",
    "Garbage collection workers have not come to {place} {temp}. Rats are appearing.",
    "The sewer line near {place} is blocked causing overflow on the main road. {urgency}.",
    "Garbage dumping near {place} is happening illegally and nobody is stopping it.",
    "The community dustbin near {place} is overflowing {temp}. {urgency}.",
    "Drainage channel near {place} has not been cleaned {temp}. Stagnant water everywhere.",
    "Open defecation is happening near the park in {place} every morning.",
    "The storm drain near {place} is clogged causing flooding during light rain.",
    "Garbage collection truck near {place} is coming once a week instead of daily.",
    "Dead animal carcass lying on the road near {place} {temp}. Extremely unhygienic.",
    "Sewer overflow from the drain near {place} is entering residential buildings.",
    "Garbage near {place} {street} has been burning for two days. Air is polluted.",
    "Manhole cover near {place} is broken and open. Very dangerous especially at night.",
    "Waste dumped near {place} flyover is attracting stray dogs and creating hazard.",
    "The nullah near {place} has not been desilted {temp}. Flooding is certain this monsoon.",
    "Garbage van in {place} colony is coming only every 3 days, waste is building up.",
    "Drain near {place} is choked with plastic waste. Sewage spilling on footpath.",
    "Public toilet near {place} is broken and overflowing {temp}. {urgency}.",
    "Loose garbage and plastic near {place} is blocking the roadside drain.",
]

TRAFFIC_TEMPLATES = [
    "The traffic signal near {place} has been blinking yellow {temp}. {urgency}.",
    "There is no traffic signal at the junction near {place}. Accidents are happening.",
    "The traffic signal near {place} is not working {temp}. Huge traffic jams forming.",
    "Illegal parking near {place} {street} is blocking traffic flow completely.",
    "There is no traffic control at {place} during peak hours. Chaos every morning.",
    "The signal near {place} has been blinking yellow for three days, it is chaos at peak hour.",
    "Vehicles are parked on both sides of the road near {place}, blocking traffic.",
    "No signal board at the intersection near {place}. Drivers don't know right of way.",
    "Traffic signal timing near {place} is wrong. Green for 5 seconds is too short.",
    "Heavy vehicles are parked on the main road near {place} {temp} causing congestion.",
    "The pedestrian signal near {place} is not working {temp}. Pedestrians at risk.",
    "Road rage incidents near {place} junction are increasing due to no signals.",
    "Wrong side driving near {place} is rampant. No enforcement whatsoever.",
    "Auto rickshaws are blocking the entire road near {place} {street} every evening.",
    "Trucks are parked illegally overnight on {place} {street}. {urgency}.",
    "The traffic signal at {place} crossroad has been dark {temp}. Manual control needed.",
    "School zone near {place} has no speed limits enforced. Children are in danger. {urgency}.",
    "The countdown timer on traffic signal near {place} is not working {temp}.",
    "Encroachment on road near {place} has reduced it to one lane causing traffic.",
    "Night time racing of vehicles near {place} is a serious safety issue. {urgency}.",
    "Footpath near {place} occupied by vendors forcing pedestrians on the road.",
    "No zebra crossing near {place} school. Children risk their life daily. {urgency}.",
    "Traffic signal poles near {place} are damaged and wires are hanging out.",
    "U-turn restriction sign near {place} has fallen down {temp}. Causing confusion.",
    "Speeding vehicles on {place} {street} are a constant threat. Need speed breaker.",
    "Three-wheeler goods vehicles parked on footpath near {place} blocking pedestrians.",
    "Bus stop near {place} has no shelter. Buses also stop in middle of road causing jam.",
    "No road marking or lane discipline near {place}. Total traffic disorder.",
]

PARKS_TEMPLATES = [
    "The park near {place} has broken swings and slides. Children are getting injured.",
    "Encroachment on the park in {place} is reducing the open space for residents.",
    "The park near {place} has not been maintained {temp}. Grass is overgrown.",
    "Stray dogs have occupied the park near {place}. Children cannot play safely.",
    "The park in {place} has broken benches and no lights. Unsafe at night.",
    "Garden area near {place} is being used for garbage dumping {temp}. {urgency}.",
    "The walking track in {place} park has broken tiles and is unsafe for walkers.",
    "Park near {place} has a broken compound wall. Outsiders are misusing it at night.",
    "Children's play equipment in {place} park is rusting and dangerous {temp}.",
    "The fountain in {place} park has not been working {temp}. It is a public attraction.",
    "Trees in {place} park have not been trimmed {temp}. Branches are a hazard.",
    "Park near {place} is being used for open drinking at night. Families cannot visit.",
    "The park lights in {place} have been broken {temp}. Park is unsafe after dark.",
    "Overgrown bushes in {place} park are becoming shelter for antisocial elements.",
    "No drinking water facility in {place} park. Residents request a water tap.",
    "The park gate near {place} is broken and park is not being locked at night.",
    "Joggers track in {place} park is full of potholes and cracks.",
    "Park near {place} has no dustbins and waste is littered everywhere.",
    "The garden near {place} has dead plants that have not been replaced {temp}.",
    "Park in {place} has an open drainage ditch inside it which is dangerous for children.",
]

# ── Assembly ─────────────────────────────────────────────────────────────────

DEPARTMENTS = {
    "Roads": (ROADS_TEMPLATES, 165),
    "Water": (WATER_TEMPLATES, 145),
    "Electricity": (ELECTRICITY_TEMPLATES, 145),
    "Sanitation": (SANITATION_TEMPLATES, 140),
    "Traffic": (TRAFFIC_TEMPLATES, 130),
    "Parks": (PARKS_TEMPLATES, 75),
}


def render(template: str) -> str:
    place = rplace()
    return (
        template
        .replace("{place}", place)
        .replace("{street}", random.choice(STREET_TYPES))
        .replace("{temp}", rtemp())
        .replace("{urgency}", rurgency())
    )


def generate_complaints():
    rows = []
    complaint_id = 1
    for dept, (templates, count) in DEPARTMENTS.items():
        generated = 0
        while generated < count:
            t = random.choice(templates)
            text = render(t)
            rows.append({
                "id": complaint_id,
                "text": text,
                "department": dept,
                "source": "synthetic",
            })
            complaint_id += 1
            generated += 1
    random.shuffle(rows)
    # re-assign ids after shuffle
    for i, r in enumerate(rows, 1):
        r["id"] = i
    return rows


def main():
    os.makedirs("data/raw", exist_ok=True)
    rows = generate_complaints()
    out_path = "data/raw/complaints_raw.csv"
    fieldnames = ["id", "text", "department", "source"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    from collections import Counter
    counts = Counter(r["department"] for r in rows)
    print(f"Saved {len(rows)} complaints to {out_path}")
    print("\nClass distribution:")
    for dept, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(rows) * 100
        print(f"  {dept:<15} {cnt:>4}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
