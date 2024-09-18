This page lists HTRFlow's built-in pipeline steps.


## Base step
::: pipeline.steps.PipelineStep
    options:
        heading_level: 3

## Pre-processing steps
::: pipeline.steps.ProcessImages
    options:
        heading_level: 3

::: pipeline.steps.Binarization
    options:
        heading_level: 3


## Inference steps
::: pipeline.steps.Inference
    options:
        heading_level: 3

::: pipeline.steps.Segmentation
    options:
        heading_level: 3

::: pipeline.steps.TextRecognition
    options:
        heading_level: 3


## Post-processing steps
::: pipeline.steps.Prune
    options:
        heading_level: 3

::: pipeline.steps.RemoveLowTextConfidencePages
    options:
        heading_level: 3

::: pipeline.steps.RemoveLowTextConfidenceRegions
    options:
        heading_level: 3

::: pipeline.steps.RemoveLowTextConfidenceLines
    options:
        heading_level: 3

::: pipeline.steps.ReadingOrderMarginalia
    options:
        heading_level: 3

::: pipeline.steps.WordSegmentation
    options:
        heading_level: 3

## Export steps
::: pipeline.steps.Export
    options:
        heading_level: 3

::: pipeline.steps.ExportImages
    options:
        heading_level: 3

## Misc
::: pipeline.steps.Break
    options:
        heading_level: 3

::: pipeline.steps.ImportSegmentation
    options:
        heading_level: 3
