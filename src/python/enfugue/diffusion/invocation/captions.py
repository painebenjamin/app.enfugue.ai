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

        with pipeline.caption_upsampler.upsampler(safe=pipeline.safe) as sampler:
            # Call task callback if set
            if task_callback is not None:
                task_callback("Upsampling captions")
            # Iterate over prompts
            for i, prompt in enumerate(self.prompts):
                prompt_results: List[str] = []
                result_times: List[float] = []
                # Iterate over captions per prompt
                for j in range(self.num_results_per_prompt):
                    start = datetime.now()
                    prompt_results.append(sampler(prompt)) # type: ignore[arg-type]
                    result_times.append((datetime.now() - start).total_seconds())
                    # Call progress callback if set
                    if progress_callback is not None:
                        progress_callback(
                            (i * self.num_results_per_prompt) + j + 1,
                            num_results,
                            sum(result_times) / len(result_times)
                        )
                # If result callback exists, call it
                if result_callback is not None and i < num_prompts - 1:
                    result_callback(prompt_results)
                # Add these results to the list of overall results
                all_results.append(prompt_results)
        return all_results

    def serialize(self) -> Dict[str, Any]:
        """
        Returns the invocation as a dict
        """
        return asdict(self)
