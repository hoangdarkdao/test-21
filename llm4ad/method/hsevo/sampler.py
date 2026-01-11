from __future__ import annotations

import re
from typing import Tuple, Optional

from ...base import LLM, Function, Program, TextFunctionProgramConverter, SampleTrimmer


class HSEvoSampler:
    """
    Sampler dành riêng cho HSEvo, tương thích với cấu trúc prompt và các loại generation khác nhau
    (initial, crossover, mutation, reflection, harmony search).
    
    Tất cả các phương thức đều trả về code sạch và thought (nếu có), đồng thời hỗ trợ
    việc extract từ response của LLM theo định dạng mà HSEvoPrompt yêu cầu.
    """

    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    # ===================================================================
    # Common utilities
    # ===================================================================
    @staticmethod
    def _extract_code_block(response: str) -> str:
        """Extract nội dung bên trong ```python ... ``` nếu có, иначе trả về toàn bộ response."""
        pattern = r"```python\s*(.*?)\s*```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Một số trường hợp LLM chỉ viết ``` mà không có python
        pattern2 = r"```\s*(.*?)\s*```"
        match2 = re.search(pattern2, response, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        return response.strip()

    @staticmethod
    def _extract_thought(response: str) -> Optional[str]:
        """
        HSEvo không yêu cầu thought trong {}, nhưng một số prompt có thể tự nhiên
        sinh ra phần mô tả trước code. Ta sẽ lấy phần text trước code block làm thought.
        """
        code = HSEvoSampler._extract_code_block(response)
        # Phần trước code block
        thought = response[: response.find(code)] if code in response else response
        thought = thought.strip()
        if thought == "":
            return None
        return thought

    def _sample_to_function(self, response: str) -> Tuple[Optional[str], Optional[Function]]:
        """Chuyển response thành thought và Function."""
        code = self._extract_code_block(response)
        thought = self._extract_thought(response)

        # Trim preface và convert thành Function
        trimmed_code = SampleTrimmer.trim_preface_of_function(code)
        function = SampleTrimmer.sample_to_function(trimmed_code, self._template_program)

        if function is not None:
            # Lưu toàn bộ code gốc để sau này tính syntax match, v.v.
            program = SampleTrimmer.sample_to_program(trimmed_code, self._template_program)
            function.entire_code = str(program) if program else trimmed_code

        return thought, function

    # ===================================================================
    # Generator (initial, crossover, mutation)
    # ===================================================================
    def get_thought_and_function(self, prompt: str, scientist_persona: str = "") -> Tuple[Optional[str], Optional[Function]]:
        """
        Dùng cho các operation sinh code chính: initial, crossover, mutation.
        Prompt đã được format sẵn (bao gồm system nếu cần).
        """
        # System prompt được nhúng trực tiếp trong HSEvoPrompt, nên chỉ cần gửi user prompt
        response = self.llm.draw_sample(prompt, temperature=0.7)  # temperature có thể config sau
        return self._sample_to_function(response)

    def get_thought_and_function_with_response(self, prompt: str, scientist_persona: str = "") -> Tuple[Optional[str], Optional[Function], str]:
        """
        Phiên bản trả thêm raw response để debug (giống MEoH).
        """
        response = self.llm.draw_sample(prompt, temperature=0.7)
        thought, func = self._sample_to_function(response)
        return thought, func, response

    # ===================================================================
    # Flash Reflection
    # ===================================================================
    def get_flash_reflection_response(self, prompt: str) -> str:
        """
        Flash reflection không cần code, chỉ cần text phân tích.
        Prompt đã bao gồm hướng dẫn format **Analysis:** và **Experience:**.
        """
        return self.llm.draw_sample(prompt, temperature=0.0)  # reflection nên deterministic

    # ===================================================================
    # Comprehensive Reflection
    # ===================================================================
    def get_comprehensive_reflection_response(self, prompt: str) -> str:
        """
        Comprehensive reflection trả về text ngắn gọn (<100 words) với 4 bullet points.
        """
        return self.llm.draw_sample(prompt, temperature=0.0)

    # ===================================================================
    # Harmony Search (parameter extraction)
    # ===================================================================
    def get_harmony_search_response(self, prompt: str) -> str:
        """
        Harmony search yêu cầu 2 phần:
          - Part 1: code với parameters mặc định
          - Part 2: parameter_ranges dict
        Trả về toàn bộ response để OptimizedHarmonySearch tự parse.
        """
        return self.llm.draw_sample(prompt, temperature=0.0)