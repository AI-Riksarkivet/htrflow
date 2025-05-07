import numpy as np
from transformers import PreTrainedTokenizerBase, VisionEncoderDecoderModel


class ConfidenceMixin:

    model: VisionEncoderDecoderModel
    processor: PreTrainedTokenizerBase

    def compute_transition_scores(self, *args, **kwargs):
        """
        This method delegates to `model.compute_transition_scores` or
        `model.decoder.compute_transition_scores`, depending on model
        architecture.
        """
        pass

    def compute_sequence_confidence_score(self, outputs):
        """Compute normalized prediction score for each output sequence

        This function computes the normalized sequence scores from the output.
        (Contrary to sequence_scores, which returns unnormalized scores)
        It follows example #1 found here:
        https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075
        """
        is_beam_search = hasattr(outputs, "beam_indices")
        transition_scores = self._compute_transition_scores(outputs).cpu()
        length_penalty = self.model.generation_config.length_penalty if is_beam_search else 1.0
        output_length = np.sum(transition_scores.numpy() < 0, axis=1)
        scores = transition_scores.sum(axis=1) / (output_length**length_penalty)
        return np.exp(scores).tolist()

    def compute_confidence_per_token(self, outputs) -> list[tuple[str, float]]:
        """
        Compute per-token confidence

        Returns the model's transition scores, per token, as a list of
        (token, score) tuples.
        """
        transition_scores = self._compute_transition_scores(outputs)
        transition_scores = np.exp(transition_scores.cpu()).tolist()
        result = []
        for sequence, scores in zip(outputs.sequences, transition_scores):
            prompt_end_index = len(sequence) - len(scores)
            tokens = self.processor.batch_decode(sequence, skip_special_tokens=True)
            pairs = zip(tokens[prompt_end_index:], scores)
            result.append([pair for pair in pairs if pair[0]])
        return result

    def _compute_transition_scores(self, outputs):
        beam_indices = getattr(outputs, "beam_indices", None)
        transition_scores = self.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            beam_indices=beam_indices,
            normalize_logits=True,
        )
        prompt_end_index = outputs.sequences.shape[1] - transition_scores.shape[1]
        sequences = outputs.sequences[:, prompt_end_index:]
        # In output from greedy decoding, padding tokens have a transition score
        # of negative infinity. To "hide" them from the score computation
        # they are set to 0 instead.
        transition_scores[sequences == self.model.generation_config.pad_token_id] = 0
        return transition_scores
