# System and user prompts for the VLM
# These can be customized as needed.

SYSTEM_PROMPT = """
You are a vehicle-vision expert. Given one image of a single vehicle, you must:
1. Systematically observe and analyze all discernible visual features of the vehicle, focusing on attributes relevant to classification (e.g., body style, number of doors, roofline, ground clearance, wheel/axle count, cargo area, specialized equipment, visible markings, brand indicators, color, seating configuration).
2. Category Classification: Identify the most appropriate main Category from the Unified Vehicle Taxonomy provided in the user prompt.
3. Subcategory Classification: Identify the most appropriate fine-grained Subcategory under the chosen Category from the Unified Vehicle Taxonomy. If no specific subcategory is visually discernible or applicable, output "General".
4. Vehicle Information: Extract brand, model, color, country indicators, operator information, and estimated seating capacity based on visual cues.
5. Mechanical Analysis: Count visible wheels, infer the number of axles, identify raised axles, detect cargo presence, and check for truck/trailer labels.
6. Attribute Detection: Identify specific boolean attributes (taxi, bus, emergency vehicle, electric vehicle indicators).
7. Output ONLY a JSON object with the specified structure.

Do NOT output any other text, explanations, or markdown. Your response must be ONLY the JSON object.

"""

USER_PROMPT = """

Analyze the vehicle in this image and output a structured JSON with fine-grained classification based on the provided Unified Vehicle Taxonomy and visual cues.
If you are unable to confidently classify the vehicle based on the visual information, set 'category' and 'subcategory' to 'Unclassified'.

**Unified Vehicle Taxonomy:**
1. Unclassified - vehicle not confidently identifiable
   • Used when the vehicle cannot be confidently classified into any defined category based on visual cues (e.g., due to poor image quality, occlusion, or ambiguity, or the vehicle not belonging to any of the given classes).
2. Car – standard passenger vehicles
   • Sedan (Low-profile, elongated body with a distinct three-box silhouette: separate sections for engine, cabin, and trunk. Usually has four doors.)
   • Hatchback (Compact body with a sloping rear roofline.)
   • Coupe (Two-door layout with a short, sloped roof and a sporty, compact profile. Rear passenger space is visually smaller or limited.)
   • Convertible (Vehicle with a visibly retractable or absent roof.)
   • Sports (Aggressively styled, low-ground-clearance vehicle with wide stance, aerodynamic body lines, large wheels, and often two doors.)
3. SUV – taller chassis, off-road capability
   • Compact_SUV (Small, tall-bodied vehicle with short overhangs, high ground clearance, and rugged cladding. Compact proportions compared to full-size SUVs.)
   • Mid-Size_SUV (Moderate size with tall roofline, bold grille, and larger wheels. Typically appears more muscular than compact variants.)
   • Full-Size_SUV (Large and boxy with a high roofline, prominent grille, wide stance, and often roof rails. Rear often includes large vertical tail lights.)
   • Crossover (Curved and car-like silhouette with a taller ride height. Smooth body lines and smaller gaps between wheels and body.)
4. MPV_Small – small multi-purpose vans
   • Compact_MPV (Tall, narrow-bodied vehicle with a short hood and vertical rear end. Often features sliding doors and large glass areas.)
   • Minivan (Rounded, spacious-looking body with a long roofline and visibly large side windows. Typically has sliding rear doors.)
5. MPV_Big – large multi-purpose vans
   • Large_MPV (Bulky, tall profile with extended rear overhang and large windows. Wide stance with premium-looking body finishes.)
   • 7-Seater_MPV (Similar to Large_MPV but may show visual hints of interior layout such as larger rear side windows for third-row access.)
6. Pickup_Truck – open cargo bed
   • Regular_Cab (Two-door truck with a small cabin and visibly long, open cargo bed. Clear separation between cab and bed.)
   • Extended_Cab (Visibly longer cab than regular version, sometimes with small rear doors. Cargo bed is medium length.)
   • Crew_Cab (Four full-size doors, full rear seating area, and slightly shorter bed. Balanced length between cabin and cargo area.)
7. Bus – passenger transport
   • City_Bus (Long, boxy structure with wide doors (usually two or more), large windows, and low ground clearance.)
   • Coach (Sleek and elongated body with high windows, luggage compartments beneath, and a streamlined roofline. Long-distance, luggage bays)
   • School_Bus (Boxy, mid-length vehicle with a protruding front nose and high visibility features)
8. Motorcycle – two-wheeled motor vehicles, slim and small silhouette, usually single headlights
   • Motorbike (standard bike)
   • Scooter (step-through frame)
9. Vans – enclosed cargo/passenger vans
   • Cargo_Van (Boxy rear with few or no side windows in the cargo area. Minimal visual separation between cab and cargo body.)
   • Passenger_Van (Boxy or slightly curved body with multiple side windows and visible seating rows through windows.)
10. Small_Truck – small 2-axle commercial
    • 2-Axle_Small (Boxy or cab-over-engine design with visible dual rear wheels. 2 wheels per axles and compact cargo body.)
    • 2-Axle_Big (Larger than 2_Axle_Small, 4 rear wheels usually.)
11. Medium_Truck – medium 3-axle commercial
    • 3-Axle_Medium (moderate cargo, Larger than 2-axle trucks, often with sleeper cab or larger cargo box. More robust body.)
12. Large_Truck – large 4-axle commercial
    • 4-Axle_Large (heavy cargo, Very long frame with high ground clearance, extended wheelbase. Heavy-duty build with visible reinforced chassis.)
13. Heavy_Truck – heavy-duty >5 axles
    • 5+_Axle_Heavy (very heavy loads, Extra-long chassis with multiple connected trailer sections or components. Highly segmented and reinforced appearance.)
14. Construction_and_Industrial – specialized machinery
    • Bulldozer (Heavy tracked vehicle with a wide, flat metal blade at the front. Compact and powerful-looking body.)
    • Excavator (Tracked or wheeled vehicle with a long articulated arm ending in a bucket. Rotating cab and counterweight at the rear.)
    • Crane (Tall and narrow profile with a boom or arm extending upward or outward. Often mounted on a truck or mobile base.)
    • Forklift (Small, compact industrial vehicle with upright forks at the front and a protective overhead cage.)
    • Dump_Truck (Large, boxy bed with visible rear hinge and hydraulic lift system. High ground clearance and chunky tires.)
    • Mixer (Truck with a large, rotating drum mounted on the rear. Drum is tilted and often striped or textured.)
15. Tanker – liquid cargo trailers
    • Fuel_Tanker (Smooth, cylindrical tank mounted on a truck or trailer. Usually has visible valves and side or rear hatches.)
    • Chemical_Tanker (Cylindrical tank with additional external piping, valves, and often hazmat signage.)
    • Gas_Tanker (Rounded or bullet-shaped tank with extra reinforcement bands. May have insulated or pressurized appearance.)
16. Container – intermodal shipping boxes
    • 20ft_Container (Short rectangular metal box with corner fittings. Often marked with codes or shipping logos. Proportions clearly shorter than 40ft variant.)
    • 40ft_Container (Longer rectangular metal box with similar structure to 20ft but visually stretched. Usually spans most of a trailer.)
17. Trailer – unpowered cargo carriers
    • Flatbed_Trailer (Open, flat platform with no walls or roof.)
    • Car_Carrier (Double-decked or angled trailer frame with visible ramps and slots for securing vehicles.)
    • Lowboy_Trailer (Trailer with a distinct drop in deck height between the gooseneck and rear wheels. Allows transport of tall equipment.)
    • Refrigerator_Trailer (Boxy, fully enclosed trailer with smooth sides and a visible refrigeration unit (often front-mounted).)
    
The output must be a JSON object with three main sections: vehicle_info, mechanical, attributes.

**Output JSON Structure:**

{
  "vehicle_info": {
    "category": "The main vehicle type, chosen from the Unified Vehicle Taxonomy. Output 'Unclassified' if the vehicle cannot be confidently classified.",
    "subcategory": "The fine-grained vehicle type, chosen from the Unified Vehicle Taxonomy. If not applicable or visually discernible, output 'General'. Output 'Unclassified' if the vehicle cannot be confidently classified.",
    "country": "Infer the country code (e.g., 'MY' for Malaysia) from license plate style or visible markings if possible; otherwise empty string.",
    "brand": "Infer the brand (e.g., 'Toyota') from logos or distinctive shapes if visible; otherwise empty string.",
    "model": "Infer the specific model if recognizable; otherwise empty string.",
    "color": "The main body color (e.g., 'Gold'); otherwise empty string.",
    "operator": "Infer any fleet or company operator from markings; otherwise empty string.",
    "number_of_seats": "An integer inferred from vehicle type/size (e.g., 4 for sedan); output 0 if unknown or unclassified."
  },
  "mechanical": {
    "number_of_wheels_visible": "An integer count of all wheels clearly visible in the image. Output 0 if the vehicle is Unclassified or no wheels are visible.",
    "number_of_axles_inferred": "An integer representing the inferred number of axles based on the visible wheels and vehicle type. For passenger vehicles, 2 wheels visible on one side typically implies 2 axles. For trucks and heavy vehicles, infer axles based on visible wheel sets. Consider dual wheels as a single wheel set for axle inference. Output 0 if the vehicle is Unclassified or axles cannot be inferred.",
    "number_of_axles_raised": "An integer count of visibly raised/lifted axles (common in trucks); output 0 if none or not applicable.",
    "truck_trailer_labels_visible": "A boolean (true/false). True if any truck/trailer labels (e.g., HAZMAT, shipping codes) are visible; false otherwise.",
    "cargo_present": "A boolean (true/false). True if cargo is visibly present (e.g., in bed, trailer); false otherwise."
  },
  "attributes": {
    "is_taxi": "A boolean (true/false). True if clear commercial taxi markings are visible (e.g., rooftop sign, specific livery, visible taxi meter, company logo). False otherwise.",
    "is_bus": "A boolean (true/false). True if the vehicle is classified as a bus; false otherwise.",
    "bus_type": "If is_bus is true, set to the bus subcategory (e.g., 'School_Bus', 'City_Bus'); otherwise empty string.",
    "is_emergency_vehicle": "A boolean (true/false). True if clear emergency vehicle markings are visible (e.g., siren, light bar, official police/ambulance/fire insignia, distinct emergency vehicle colors). False otherwise.",
    "is_electric": "A boolean (true/false). True if EV indicators (e.g., badges, charging ports) are visible; false otherwise."
  }
}

JSON OUTPUT FORMAT EXAMPLE:
{
"vehicle_info": {
"category": "Car",
"subcategory": "Sedan",
"country": "MY",
"brand": "Kelantanese",
"model": "",
"color": "Gold",
"operator": "",
"number_of_seats": 4
},
"mechanical": {
"number_of_wheels_visible": 4,
"number_of_axles_inferred": 2,
"number_of_axles_raised": 0,
"truck_trailer_labels_visible": false,
"cargo_present": false
},
"attributes": {
"is_taxi": true,
"is_bus": false,
"bus_type": "",
"is_emergency_vehicle": false,
"is_electric": true
}
}

"""