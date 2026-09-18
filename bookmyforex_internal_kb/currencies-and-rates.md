# Currencies, Exchange Rates & Rate Handling

## Product-specific rates
Do not assume one BookMyForex rate applies across products. Public pages explicitly distinguish:
- Currency notes / cash
- Forex Card
- Wire transfer / remittance
- Other supported forex instruments

Physical cash can carry different pricing because of stocking, movement, insurance and handling. Remittance is electronic and may have a different margin.

## Live-rate handling
Exchange rates are dynamic during market hours and can remain static across weekends/public holidays until markets reopen. For internal RAG:
- Do **not** store a numerical exchange rate as durable knowledge.
- Fetch/display the current website/app rate when a customer asks "today's rate".
- Preserve product type, buy/sell direction, city and currency because each can affect the quote.

## Currency availability
BookMyForex supports major and exotic currencies, but availability varies by location/product. Currency notes and denominations are subject to stock.

## Zero-markup wording
"Zero markup/interbank" is not a blanket promise across all BookMyForex products. Current public terms restrict it by card variant, currencies, city/order amount, market hours and offer conditions. Cross-currency use and refunds/unloads can be excluded.

## Rate lock
- Retail currency-exchange pages describe Rate Freeze for a limited period under deposit/booking terms.
- Money-transfer pages advertise rate lock-in up to 3 working days.
- Trade-remittance page advertises SPOT/forward facilities and longer lock periods depending on product.

## Programmatic rate pages
The site contains many currency × city pages. For RAG, retain the rules and fetch live rates dynamically rather than embedding hundreds of near-identical city pages.

## Sources
- https://www.bookmyforex.com/currencyExchange.htm
- https://www.bookmyforex.com/money-transfer/exchange-rates/
- https://www.bookmyforex.com/terms-of-use/
- https://www.bookmyforex.com/trade-remittance/
