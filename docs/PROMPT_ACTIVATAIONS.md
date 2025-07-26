# WanVideo Activation Editor Test Prompts

This document contains carefully crafted test prompts for the WanVideo Activation Editor. Each test case is designed to demonstrate different aspects of block-level prompt injection, following the exact style of WAN's training data.

## How to Use These Tests

1. Copy the main prompt into a WanVideoTextEncode node
2. Copy the injection prompt into a second WanVideoTextEncode node
3. Use WanVideoBlockActivationBuilder with the specified preset/pattern
4. Start with injection_strength around 0.3-0.5
5. Compare results with and without activation editing

## Test Categories

### 1. Environmental Transformation Tests
These test how well the model can blend different environments and atmospheres.

#### Test 1.1: Office → Volcanic
**Purpose**: Transform a modern office into a volcanic environment at high semantic level  
**Block Pattern**: `late_blocks`  
**Expected**: Office scene with lava-like lighting, heat distortion, volcanic atmosphere

**Main Prompt**:
```
A woman sits at a wooden desk in a modern office, typing on a silver laptop. She wears a white blouse and black-framed glasses. The desk is organized with a potted succulent, a ceramic coffee mug, and neatly stacked papers. Behind her, floor-to-ceiling windows reveal a cityscape with glass skyscrapers reflecting afternoon sunlight. The office has minimalist decor with white walls and a single abstract painting. Natural light floods the space, creating soft shadows across the desk surface.
```

**Injection Prompt**:
```
Molten lava flows down a volcanic mountainside in slow, mesmerizing rivers. The glowing orange-red magma radiates intense heat, causing the air above to shimmer and distort. Black volcanic rock cools and hardens at the edges of the flow, creating intricate patterns. Smoke and steam rise from where the lava meets patches of vegetation. The night sky glows red from the reflected light of the eruption. Sparks and embers float through the air like fireflies.
```

#### Test 1.2: Living Room → Blizzard
**Purpose**: Inject harsh weather into cozy interior  
**Block Pattern**: `mid_blocks`  
**Expected**: Interior with snow/frost effects, cold lighting, storm elements

**Main Prompt**:
```
A cozy living room scene shows a family gathered around a stone fireplace. Two children sit on a plush carpet, playing a board game. Their parents relax on a leather sofa, the mother reading a book while the father watches the children. The fire crackles warmly, casting dancing shadows on wood-paneled walls. Family photos line the mantelpiece. A large dog sleeps peacefully near the hearth. The room has warm lighting from table lamps, creating an intimate atmosphere.
```

**Injection Prompt**:
```
A powerful blizzard engulfs a mountain landscape. Snow falls heavily, driven by fierce winds that create near whiteout conditions. Ice crystals swirl violently through the air, accumulating in deep drifts. Pine trees bend under the weight of snow and force of the gale. Visibility is reduced to mere meters as the storm rages. The howling wind creates an overwhelming soundscape. Everything is monochrome white and gray, with occasional dark shapes of trees emerging from the maelstrom.
```

#### Test 1.3: Hospital → Enchanted Forest
**Purpose**: Transform clinical space into magical environment  
**Block Pattern**: `late_blocks`  
**Expected**: Medical setting with bioluminescent, magical qualities

**Main Prompt**:
```
A modern hospital room is equipped with advanced medical technology. A patient rests in an adjustable bed while a nurse checks vital signs on a digital monitor. IV drips hang from metal poles. The room has white walls and a large window overlooking a parking lot. Medical equipment beeps softly at regular intervals. Everything is sterile and organized. A visitor sits in a blue vinyl chair beside the bed. Fluorescent lights create even, clinical illumination. Hand sanitizer dispensers are mounted on walls.
```

**Injection Prompt**:
```
An enchanted forest glows with bioluminescent magic at twilight. Ancient trees have twisted trunks covered in glowing moss that pulses with soft blue light. Fireflies dance between branches like living stars. Mushroom circles emit a gentle golden radiance from the forest floor. A crystal-clear brook babbles over smooth stones that seem to glow from within. Mystical fog drifts between the trees, carrying sparkles of light. Flowers open to reveal luminescent centers. The air shimmers with barely visible magical energy.
```

### 2. Activity Contrast Tests
These test blending contradictory activities or energy levels.

#### Test 2.1: Gardening → DJ Performance
**Purpose**: Mix peaceful activity with high-energy scene  
**Block Pattern**: `alternating`  
**Expected**: Surreal mix of calm gardening with nightclub energy

**Main Prompt**:
```
An elderly man tends to his garden in the early morning light. He kneels beside rows of tomato plants, carefully pruning leaves with small garden shears. His weathered hands move with practiced precision. He wears a straw hat, plaid shirt, and worn denim overalls. The garden is lush with vegetables and flowers, surrounded by a white picket fence. Dewdrops glisten on leaves, and a wooden shed stands in the background. Birds chirp softly as the peaceful scene unfolds.
```

**Injection Prompt**:
```
A DJ performs at a packed nightclub, hands moving rapidly across turntables and mixing board. Strobe lights flash in sync with the pounding bass, creating a frenetic visual rhythm. The crowd dances energetically, hands raised toward the ceiling. Laser beams cut through fog machine haze in green and blue patterns. The DJ wears headphones and a black t-shirt, completely absorbed in mixing tracks. LED screens behind the booth display psychedelic visuals that pulse with the music.
```

#### Test 2.2: Library Study → Carnival
**Purpose**: Inject celebration into quiet study  
**Block Pattern**: `early_blocks`  
**Expected**: Library with carnival colors, festive textures

**Main Prompt**:
```
A university library reading room maintains perfect silence. Students sit at long wooden tables, surrounded by open textbooks and laptops. Tall windows with heavy curtains filter afternoon light. Floor-to-ceiling bookshelves create corridors of knowledge. A librarian organizes returned books on a cart. The atmosphere is focused and scholarly. Ornate architectural details include carved wooden panels and a coffered ceiling. Some students wear headphones while others take handwritten notes. The only sounds are occasional page turns and quiet footsteps.
```

**Injection Prompt**:
```
A Brazilian carnival parade explodes with color and movement down a crowded street. Dancers in elaborate feathered costumes perform samba moves on moving floats. Their outfits sparkle with sequins and gems in electric blues, hot pinks, and golden yellows. Drummers maintain infectious rhythms that echo off buildings. Confetti cannons shoot streams of paper into the air. The crowd dances along, waving flags and wearing masks. The energy is euphoric and contagious. Speakers blast samba music that can be heard blocks away.
```

#### Test 2.3: Yoga Class → Storm Chase
**Purpose**: Blend serenity with extreme weather pursuit  
**Block Pattern**: `custom: 1111111111000000000000000000001111111111`  
**Expected**: Peaceful yoga with storm energy at beginning/end

**Main Prompt**:
```
A serene yoga class takes place in a sunlit studio. Practitioners hold warrior pose on purple mats, their breathing synchronized and calm. The instructor demonstrates proper alignment, speaking in soothing tones. Large windows reveal a garden view. The hardwood floor is polished and clean. Soft instrumental music plays from hidden speakers. Everyone wears comfortable athletic wear in muted colors. Blocks and straps are neatly arranged along one wall. The energy is peaceful and focused. Afternoon light creates geometric patterns through the windows.
```

**Injection Prompt**:
```
Storm chasers pursue a massive tornado across Oklahoma plains. Their reinforced vehicle speeds down a dirt road as the funnel cloud touches down in the distance. Lightning flashes illuminate the dark green sky. The tornado kicks up debris, creating a debris cloud at its base. Scientific instruments on the vehicle's roof spin wildly in the wind. Radio chatter coordinates with other chase teams. The landscape is flat farmland with isolated structures. Rain begins pelting the windshield. The raw power of nature dominates everything.
```

### 3. Material/Texture Transformation Tests
These test how different materials and textures blend.

#### Test 3.1: Tech Lab → Rainforest
**Purpose**: Blend synthetic with organic  
**Block Pattern**: `early_blocks`  
**Expected**: Lab equipment with organic textures, natural materials

**Main Prompt**:
```
A state-of-the-art robotics laboratory buzzes with activity. Robotic arms perform precise assembly tasks on a production line. LED indicators blink in various colors on control panels. Engineers in white lab coats monitor computer screens displaying complex data visualizations. The space is brightly lit with fluorescent lighting, revealing clean white surfaces and organized tool stations. Cables and wires run along designated channels. The air hums with the sound of servos and cooling fans.
```

**Injection Prompt**:
```
An ancient rainforest teems with life in the humid afternoon. Massive tree trunks rise like cathedral pillars, their bark covered in moss and climbing vines. Exotic birds with brilliant plumage call from the canopy. A small stream winds between root systems, its water crystal clear. Butterflies dance through shafts of filtered sunlight. The forest floor is dense with ferns and fallen leaves. Mist hangs in the air, creating an ethereal quality. The sounds of insects and distant monkey calls fill the space.
```

#### Test 3.2: Sushi Bar → Assembly Line
**Purpose**: Mix culinary precision with industrial automation  
**Block Pattern**: `mid_blocks`  
**Expected**: Culinary scene with mechanical, industrial qualities

**Main Prompt**:
```
A master sushi chef prepares omakase at an intimate counter. His knife moves with precision, slicing fresh tuna into perfect pieces. The bamboo cutting board shows years of use. He places each piece on handmade ceramic plates with minimal garnish. Behind him, a refrigerated case displays today's selection: salmon, yellowtail, and sea urchin. The counter is blonde wood, polished to a high sheen. Only eight seats face the chef. Soft lighting highlights the food's natural colors. The atmosphere is quiet and reverent.
```

**Injection Prompt**:
```
An automated car assembly line operates with mechanical precision. Robotic arms weld chassis components in showers of sparks. The production line moves steadily, carrying partial vehicles through different stations. Overhead conveyor systems deliver parts exactly when needed. Workers in safety gear perform quality checks at specific points. The factory floor is vast and organized. Yellow safety lines mark walkways. The sound of pneumatic tools and electric motors creates an industrial symphony. Everything moves in coordinated efficiency.
```

### 4. Atmosphere/Mood Transformation Tests
These test emotional and atmospheric blending.

#### Test 4.1: Wedding → Rock Concert
**Purpose**: Transform elegant ceremony into raw energy  
**Block Pattern**: `custom: 1010101010101010101010101010101010101010`  
**Expected**: Formal event with rock concert elements alternating throughout

**Main Prompt**:
```
An elegant outdoor wedding ceremony takes place in a rose garden. The bride walks down a petal-strewn aisle in a flowing white gown with intricate lace details. The groom waits at an altar decorated with white flowers and greenery. Guests sit in white folding chairs arranged in neat rows. A string quartet plays softly from the side. Golden hour light bathes everything in warm tones. The setting is peaceful and romantic, with manicured hedges creating natural walls. Everyone is dressed formally.
```

**Injection Prompt**:
```
A heavy metal concert reaches its climax in a packed arena. The lead guitarist shreds a solo while pyrotechnics explode behind the band. The crowd forms a massive mosh pit, bodies colliding in controlled chaos. Stage lights strobe in red and white, cutting through smoke effects. The drummer pounds relentlessly, hair flying with each strike. Marshall stacks line the stage edges. Fans throw devil horns hand signs. The energy is raw and electric, with the bass so loud it vibrates through bodies.
```

#### Test 4.2: Beach Vacation → Arctic Expedition
**Purpose**: Transform tropical warmth to polar cold  
**Block Pattern**: `second_half`  
**Expected**: Beach scene that becomes increasingly arctic

**Main Prompt**:
```
A family enjoys a perfect beach day under clear blue skies. Children build sandcastles near the water's edge while parents relax under a striped umbrella. Turquoise waves gently lap at the shore. Seagulls circle overhead. Beach towels in bright colors dot the white sand. A vendor walks by selling cold drinks from a cooler. Palm trees provide patches of shade. In the distance, sailboats drift lazily on the horizon. The air smells of salt and sunscreen. Everyone wears swimsuits and sunglasses.
```

**Injection Prompt**:
```
An arctic research station sits isolated on a vast ice shelf. Scientists in heavy parkas and goggles check weather instruments that are coated with frost. The landscape is blindingly white, stretching endlessly under a pale sun. Icy wind whips snow crystals across the surface. A tracked vehicle sits nearby, equipped for polar conditions. Radio antennas and satellite dishes point skyward. The station's red buildings provide the only color against the monochrome environment. Aurora borealis hints green on the horizon.
```

### 5. Scene Complexity Tests
These test how complex multi-element scenes blend.

#### Test 5.1: Urban Night → Coral Reef
**Purpose**: Blend city lights with underwater bioluminescence  
**Block Pattern**: `sparse`  
**Expected**: Urban scene with subtle underwater qualities

**Main Prompt**:
```
A busy downtown intersection at night pulses with urban energy. Neon signs advertise restaurants and shops in vibrant colors. Pedestrians cross streets while cars wait at red lights, their headlights creating light trails. Steam rises from a subway grate. A food truck serves customers on the corner. Tall buildings frame the scene, their windows glowing with interior lights. The wet pavement reflects the colorful urban lighting, creating a mirror-like effect. Police sirens echo in the distance.
```

**Injection Prompt**:
```
A vibrant coral reef ecosystem thrives in crystal-clear tropical waters. Schools of yellow tang fish swim in perfect synchronization past brain coral formations. A sea turtle glides gracefully over the reef, its shell catching dappled sunlight from above. Purple sea fans sway in the gentle current. Clownfish dart in and out of anemone homes. The water has a brilliant turquoise color, with visibility extending far into the blue distance. Small bubbles rise from hidden crevices where crabs hide.
```

#### Test 5.2: Fashion Show → Archaeological Dig
**Purpose**: Mix glamour with ancient discovery  
**Block Pattern**: `alternating`  
**Expected**: High fashion with archaeological elements throughout

**Main Prompt**:
```
A high-fashion runway show captivates an exclusive audience. A model struts confidently down the catwalk wearing an avant-garde dress made of metallic fabric that catches the spotlight. Photographers line the runway's edge, cameras flashing continuously. The backdrop is minimalist white with the designer's logo. Other models wait backstage, being fitted with accessories by stylists. The audience consists of fashion editors and celebrities in the front row. Electronic music pulses through the venue. The atmosphere is glamorous and cutting-edge.
```

**Injection Prompt**:
```
An archaeological excavation site reveals ancient civilization remains. Archaeologists carefully brush dirt from pottery shards using small tools. The dig is organized in a grid pattern with string markers. Layers of earth show different historical periods. A partially uncovered stone wall suggests a larger structure below. Sifting screens separate artifacts from soil. Field notebooks document each find's location. The site is under a protective canopy. Crates hold catalogued items. The work is meticulous and patient, with history emerging grain by grain.
```

### 6. Special Effect Tests
These test specific visual effects and transformations.

#### Test 6.1: Kitchen → Steel Foundry
**Purpose**: Inject industrial heat and metal into domestic scene  
**Block Pattern**: `custom: 0000111111000000000000000000111111000000`  
**Expected**: Kitchen with molten metal qualities in specific sections

**Main Prompt**:
```
A grandmother prepares Sunday dinner in her traditional kitchen. She stirs a large pot of soup on the gas stove, steam rising fragrantly. The counter is covered with fresh ingredients: chopped vegetables, herbs, and homemade pasta. Copper pots hang from hooks above. Sunlight streams through lace curtains, illuminating flour dust in the air. She wears a floral apron over her dress. The kitchen has vintage tiles and wooden cabinets filled with china. A cat sleeps on a chair nearby.
```

**Injection Prompt**:
```
A massive steel foundry operates at full capacity. Molten metal pours from enormous crucibles into molds, glowing white-hot and sending up showers of sparks. Workers in protective gear and helmets monitor the process from safe distances. Overhead cranes move huge metal components. The industrial space is filled with catwalks, pipes, and heavy machinery. Heat waves distort the air. Warning lights flash yellow and red. The constant clang of metal on metal reverberates through the facility.
```

#### Test 6.2: Auto Shop → Underwater Cave
**Purpose**: Transform mechanical space into aquatic environment  
**Block Pattern**: `custom: 0000000000111111111100000000001111111111`  
**Expected**: Auto shop with underwater sections

**Main Prompt**:
```
A busy auto repair shop echoes with the sound of power tools. A mechanic slides under a raised car on a hydraulic lift, oil-stained coveralls protecting his clothes. Tool boxes on wheels hold wrenches, sockets, and diagnostic equipment. The concrete floor has various stains from years of service. Fluorescent tubes provide harsh overhead lighting. Tires are stacked against one wall. A customer waits in a small office area, watching through a window. The air smells of motor oil and rubber.
```

**Injection Prompt**:
```
An underwater cave system reveals hidden wonders to explorers. Stalactites hang from the ceiling, submerged in crystal-clear water that reflects dancing light patterns. Schools of blind cave fish navigate the darkness. A diver's light beam cuts through the water, illuminating ancient rock formations. Air bubbles rise from the diver's regulator, creating silver streams. The cave walls are covered in delicate formations built over millennia. Shafts of blue light penetrate from distant openings. The silence is profound, broken only by the sound of breathing apparatus.
```

#### Test 6.3: Pet Grooming → Space Station
**Purpose**: Add zero-gravity elements to earthbound scene  
**Block Pattern**: `sparse`  
**Expected**: Grooming salon with subtle weightless qualities

**Main Prompt**:
```
A professional pet grooming salon bustles with activity. A groomer carefully trims a poodle's fur into a show cut, scissors moving expertly. The dog stands patiently on a non-slip table. Other dogs wait in comfortable crates, some barking excitedly. The walls display before-and-after photos of groomed pets. Specialized equipment includes electric clippers, nail grinders, and various brushes. The floor has a drain for easy cleaning. Everything smells of dog shampoo. A reception area has treats and toys for sale. The atmosphere is cheerful despite occasional protests from reluctant clients.
```

**Injection Prompt**:
```
The International Space Station orbits Earth in perfect silence. Astronauts float weightlessly through modules, pushing off walls to navigate. Equipment is secured with velcro and clips to prevent floating. Through the cupola windows, Earth's blue curve contrasts with the black of space. Solar panels extend like wings, catching sunlight. Inside, LED panels provide constant illumination. Laptops display mission data while experiments run in specialized racks. Everything has a purpose and place. The constant hum of life support systems provides background white noise.
```

## Testing Strategy

### Progressive Testing
1. Start with `early_blocks` to see texture/color changes
2. Move to `mid_blocks` for pattern/structure blending
3. Try `late_blocks` for semantic/compositional changes
4. Use `alternating` or `sparse` for mixed effects

### Strength Recommendations
- **0.2-0.3**: Subtle influence, maintains main scene integrity
- **0.4-0.5**: Balanced blending, clear influence from both prompts
- **0.6-0.8**: Strong injection, may override main prompt significantly

### Custom Block Patterns
For fine control, create custom patterns targeting specific transformer depths:
- **Blocks 0-10**: Low-level features (textures, colors, basic patterns)
- **Blocks 10-20**: Mid-level features (shapes, local structures)
- **Blocks 20-30**: High-level features (objects, scene composition)
- **Blocks 30-40**: Semantic features (meaning, relationships, global structure)

## Notes
- All prompts follow WAN training data style for optimal results
- Test with and without activation for comparison
- Monitor console output to verify block activation
- Some combinations may produce unexpected artistic results