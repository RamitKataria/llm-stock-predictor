from torchtune.data import InstructTemplate
from typing import Any, Dict, Mapping, Optional

class StockNewsTemplate(InstructTemplate):
    """
    Prompt template for datasets containing news articles and stock price changes.
    """

    template = 'A news story about {company} in the {industry} industry, in the {sector} sector, with stock ticker {ticker} was posted on {url} on {datetime} with title "{title}". Here\'s the description they provided: {description}. Given this, what should the stock price change be, in percent, the next day?'

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
        datetime = sample[column_map.get("datetime", "datetime")]
        url = sample[column_map.get("url", "url")]
        title = sample[column_map.get("title", "title")]
        description = sample[column_map.get("description", "description")]
        company = sample[column_map.get("company", "company")]
        ticker = sample[column_map.get("ticker", "ticker")]
        sector = sample[column_map.get("sector", "sector")]
        industry = sample[column_map.get("industry", "industry")]
        prompt = cls.template.format(datetime=datetime, url=url, title=title, description=description, company=company, ticker=ticker, sector=sector, industry=industry)
        return prompt
