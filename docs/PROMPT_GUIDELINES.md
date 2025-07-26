# Prompt Guidelines for Effective Activation Injection

## The Problem: Similar Prompts = No Effect

If your debug output shows low "Percent changed" values (like 1.1%), your prompts are too similar. The T5 encoder is producing nearly identical embeddings, so there's nothing to inject!

## WAN Prompt Style

WanVideo was trained on detailed, descriptive captions. Write complete scene descriptions with specific details about:
- Subject appearance and actions
- Environment and setting
- Lighting and atmosphere
- Objects and their positions
- Colors and textures

## What Makes Prompts Different?

The T5 encoder looks at semantic meaning, not just words. These are TOO SIMILAR:
- "beautiful forest" vs "nice forest" (synonyms)
- "red car" vs "crimson vehicle" (same concept)
- "happy person" vs "joyful human" (same meaning)

## Effective Prompt Pairs (WAN Style)

### Environmental Opposites
**Main**: "A bustling beach scene on a bright summer day. Several people are scattered across the sandy shore, some lounging under colorful umbrellas while others play volleyball. The ocean waves are gentle, with children splashing in the shallow water. Palm trees line the beach, their fronds swaying in the warm breeze."

**Injection**: "A desolate mountain peak covered in deep snow during a blizzard. The landscape is entirely white, with harsh winds blowing snow horizontally across the frame. No vegetation is visible, only rocky outcrops occasionally breaking through the snow cover."

**Expected**: ~65-75% difference

### Activity Contrast
**Main**: "A person sitting cross-legged on a yoga mat in a serene indoor space. They are wearing comfortable exercise clothing and have their eyes closed in meditation. The room has wooden floors and large windows letting in soft morning light."

**Injection**: "A crowded rock concert venue with a band performing on stage. Bright colored stage lights flash rapidly in sync with the loud music. The crowd is densely packed, with people jumping and raising their hands."

**Expected**: ~60-70% difference

### Material/Texture Transformation
**Main**: "A modern building constructed entirely of glass and steel. The facade reflects the surrounding cityscape, creating a mirror-like effect. The structure features clean geometric lines with floor-to-ceiling windows on each level."

**Injection**: "An ancient temple carved from rough sandstone blocks. The weathered surface shows centuries of erosion, with intricate relief carvings partially worn away. Moss and vines grow in the cracks between the massive stone blocks."

**Expected**: ~55-65% difference

## Bad Examples (Too Similar)

- Main: "A forest path in springtime"  
  Injection: "A forest trail in summer"  
  Difference: ~5-10% ❌

- Main: "A woman wearing a red dress"  
  Injection: "A woman in a blue dress"  
  Difference: ~2-5% ❌

## Checking Your Prompts

Before running generation, check the debug output:
```
[WAN_DEBUG] Block 0 Blending Results:
  - Percent changed: 78.4%  ✓ GOOD - very different
  - Percent changed: 1.1%   ✗ BAD - too similar
```

Aim for > 50% difference for noticeable effects!

## Quick Tips

1. **Domain Shifts**: Photo → Painting, Modern → Ancient, Natural → Synthetic
2. **Mood Opposites**: Peaceful → Chaotic, Bright → Dark, Happy → Melancholy
3. **Environment Changes**: Indoor → Outdoor, Urban → Nature, Day → Night
4. **Style Contrasts**: Realistic → Abstract, Clean → Weathered, Minimal → Ornate

## Testing Before Generation

Use WanVideoVectorDifference node to measure embedding difference:
1. Encode both prompts with WanVideoTextEncode
2. Connect to WanVideoVectorDifference
3. Check the output statistics
4. Adjust prompts if difference < 50%