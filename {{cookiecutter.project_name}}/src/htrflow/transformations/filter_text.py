class TextRecResult:
    def __init__(self, text: str, score: Tensor):
        self.text = text
        self.score = score


class PostProcessTranscription:
    def __init__(self):
        pass

    def add_trans_to_result(result_full, result_lines):
        ind = 0

        for res in result_full:
            for i, nested_res in enumerate(res.nested_results):
                nested_res.texts = []
                for j in range(0, len(nested_res.segmentation.bboxes)):
                    nested_res.texts.append(result_lines[ind + j])
                ind += len(nested_res.segmentation.bboxes)
