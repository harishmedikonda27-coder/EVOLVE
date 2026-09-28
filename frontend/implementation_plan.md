# Implementation Plan: Synchronize Abstract Architecture with Actual Architecture

## Goal Description
The `BrainTopology.tsx` (Abstract view) currently uses hardcoded structural groupings (e.g., exactly 4 nodes for Encoder, 4 for Router) that do not reflect the actual underlying PyTorch `Linear` layers, nor does it accurately reflect deep modules. We need to refactor `BrainTopology.tsx` to read the precise PyTorch architecture (`actual_layers`) identical to `ActualArchitectureView.tsx`, but render it as a visually appealing compressed representation (SVG with representative nodes and animated bezier edges).

## Proposed Changes

### [MODIFY] `components/BrainTopology.tsx`
- **Align Layer Parsing**: Replace the hardcoded `input -> encoder -> router -> module -> head -> output` block logic with the dynamic `actual_layers` parsing logic used in `ActualArchitectureView.tsx`.
- **Representative Nodes**: For every PyTorch `Linear` layer detected in the topology, generate a corresponding SVG column of nodes. Instead of generating `N` nodes matching the true dimension (e.g. 256), generate a compressed representative count (e.g., `Math.min(actualDim, 4)` or a fixed aesthetic number like 3-4 nodes).
- **Representative Edges**: Between adjacent topological layers, generate representative bezier edges. If the layer is sparse (e.g. router to module), still generate representative connections without querying the exact million-parameter boolean mask, allowing the SVG animation to flow through the correct macroscopic structural path.
- **Styling Preservation**: Maintain the exact SVG bezier curve aesthetics, pulsating animations, and color groups (`blue` for encoder, `purple` for router/head, `emerald` for modules) so that the visual "vibe" of the Abstract view remains intact.
- **Stable Layout**: Scale the `spacingX` dynamically based on the total number of actual parsed layers to ensure the network fits beautifully within the `viewBox="0 0 100 100"`, matching the responsive nature of the current SVG.

## Verification Plan
### Automated Tests
- Run `npx tsc --noEmit` and `npm run build` to ensure no type errors.
### Manual Verification
- Review the `BrainTopology.tsx` implementation to confirm it strictly iterates over `actual_layers`.
- (User Action) The user will visually inspect that clicking the Brain properly crossfades between the Abstract and Actual views, and that the horizontal layer counts and color sequences perfectly match between the two modes.
