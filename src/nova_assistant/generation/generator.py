import re

from openai import OpenAI

from nova_assistant.generation.models import (
    GeneratedAnswer,
    GenerationRequest,
)
from nova_assistant.generation.prompt_builder import (
    build_generation_prompt,
)

DEFAULT_GENERATION_MODEL = "gpt-5.4-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 300


class OpenAIGenerator:
    """Génère une réponse citée avec la Responses API."""

    def __init__(
        self,
        model_name: str = DEFAULT_GENERATION_MODEL,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        client: OpenAI | None = None,
    ) -> None:
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens doit être strictement positif.")

        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.client = client or OpenAI(
            max_retries=1,
            timeout=30.0,
        )

    def generate(
        self,
        request: GenerationRequest,
    ) -> GeneratedAnswer:
        prompt = build_generation_prompt(request)

        response = self.client.responses.create(
            model=self.model_name,
            instructions=prompt.system_prompt,
            input=prompt.user_prompt,
            max_output_tokens=self.max_output_tokens,
            store=False,
        )

        answer = response.output_text.strip()

        if not answer:
            raise ValueError("Le modèle n’a produit aucune réponse textuelle.")

        used_source_ids = set(re.findall(r"\[(S[1-9][0-9]*)\]", answer))
        available_sources = {source.source_id: source for source in prompt.sources}
        unknown_source_ids = used_source_ids - available_sources.keys()

        if unknown_source_ids:
            unknown_list = ", ".join(sorted(unknown_source_ids))
            raise ValueError(f"Le modèle a inventé des citations : {unknown_list}.")

        if not used_source_ids:
            raise ValueError("Le modèle n’a cité aucune source.")

        citations = tuple(
            source for source in prompt.sources if source.source_id in used_source_ids
        )

        return GeneratedAnswer(
            answer=answer,
            model_name=self.model_name,
            citations=citations,
        )
