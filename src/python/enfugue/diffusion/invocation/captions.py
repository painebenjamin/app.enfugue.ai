from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, asdict

from typing import List, Optional, Callable, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from enfugue.diffusion.manager import DiffusionPipelineManager

__all__ = ["CaptionInvocation"]

@dataclass
class CaptionInvocation:
    """
    A serializable class holding all vars for getting captions
    """
    prompts: List[str] # Required
    num_results_per_prompt: int = 1

    def execute(
        self,
        pipeline: DiffusionPipelineManager,
        task_callback: Optional[Callable[[str], None]] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        result_callback: Optional[Callable[[List[str]], None]] = None,
    ) -> List[List[str]]:
        """
        This is the main interface for execution.
        """
        if task_callback is not None:
            task_callback("Preparing language pipeline")

        num_prompts = len(self.prompts)
        num_results = num_prompts * self.num_results_per_prompt
        all_results: List[List[str]] = []

        with pipeline.caption_upsampler.upsampler() as sampler:
            if task_callback is not None:
                task_callback("Upsampling captions")
            for i, prompt in enumerate(self.prompts):
                prompt_results: List[str] = []
                result_times: List[float] = []
                for j in range(self.num_results_per_prompt):
                    start = datetime.now()
                    upsampled = sampler(prompt)
                    result_times.append((datetime.now() - start).total_seconds())
                    if progress_callback is not None:
                        progress_callback(
                            (i * self.num_results_per_prompt) + j + 1,
                            num_results,
                            sum(result_times) / len(result_times)
                        )
                if result_callback is not None and i < num_prompts - 1:
                    result_callback(prompt_results)
                all_results.append(prompt_results)

        return all_results

    def serialize(self) -> Dict[str, Any]:
        """
        Returns the invocation as a dict
        """
        return asdict(self)
