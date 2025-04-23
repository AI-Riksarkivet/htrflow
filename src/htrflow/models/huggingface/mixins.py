import numpy as np
from transformers import VisionEncoderDecoderModel


class ConfidenceMixin:

    model: VisionEncoderDecoderModel

    def compute_transition_scores(self):
        pass

    def compute_sequence_confidence_score(self, outputs):
        """Compute normalized prediction score for each output sequence

        This function computes the normalized sequence scores from the output.
        (Contrary to sequence_scores, which returns unnormalized scores)
        It follows example #1 found here:
        https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
        """
        beam_indices = getattr(outputs, "beam_indices", None)
        transition_scores = self.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            beam_indices=beam_indices,
            normalize_logits=True,
        )

        is_beam_search = beam_indices is not None
        length_penalty = self.model.generation_config.length_penalty if is_beam_search else 1.0
        # In output from greedy decoding, padding tokens have a transition score
        # of negative infinity. To "hide" them from the score computation
        # they are set to 0 instead.
        transition_scores[outputs.sequences[:, 1:] == self.model.generation_config.pad_token_id] = 0
        transition_scores = transition_scores.cpu()
        output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        return np.exp(scores).tolist()
