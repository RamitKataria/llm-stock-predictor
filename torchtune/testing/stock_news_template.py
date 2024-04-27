from torchtune.data import InstructTemplate
from typing import Any, Dict, Mapping, Optional

class StockNewsTemplate(InstructTemplate):
    """
    Prompt template for datasets containing news articles and stock price changes.
    """

    template = "News: {news}\nDate: {date}\nIndustry: {industry}\nSentiment: {sentiment}\n"

    @classmethod
    def format(cls, sample: Mapping[str, Any], column_map: Optional[Dict[str, str]] = None) -> str:
        """
        Generate prompt from instruction and input.

        Args:
            sample (Mapping[str, Any]): a single data sample.
            column_map (Optional[Dict[str, str]]): a mapping from placeholder names in the template to column names.

        Returns:
            The formatted prompt.
        """
        column_map = column_map or {}
        news = sample[column_map.get("news", "News")]
        date = sample[column_map.get("date", "Date")]
        industry = sample[column_map.get("industry", "Industry")]
        sentiment = sample[column_map.get("sentiment", "Sentiment")]
        prompt = cls.template.format(news=news, date=date, industry=industry, sentiment=sentiment)
        return prompt
