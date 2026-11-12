"""Generate the synthetic Harbor Goods corpus and the question set (answer keys).

Everything here is made up: Harbor Goods, its suppliers, carriers, prices and people do not exist.

Writes, relative to the repo root:
- data/corpus/<doc_id>.md          for the 16 markdown documents
- data/corpus/<doc_id>.source.md   for the 4 documents that make_pdfs.py renders to PDF
- data/questions/questions.json    60 answer-key objects

Repeatable: no timestamps, no randomness, fixed order, UTF-8 with "\\n" line endings.
Standard library only.

    python scripts/make_corpus.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
QUESTIONS_PATH = ROOT / "data" / "questions" / "questions.json"

PAGEBREAK = "<!-- pagebreak -->"

# Each document: doc_id, title, version, effective date, pdf flag, body (starts with "# <title>").
# PDF bodies hold 2 or 3 pages separated by a line that is exactly PAGEBREAK.

DOCS: list[dict] = []


def doc(
    doc_id: str, title: str, version: str, effective: str, body: str, pdf: bool = False
) -> None:
    DOCS.append(
        {
            "doc_id": doc_id,
            "title": title,
            "version": version,
            "effective": effective,
            "pdf": pdf,
            "body": body.strip("\n"),
        }
    )


doc(
    "shipping-policy",
    "Shipping Policy",
    "2026-01",
    "2026-01-15",
    """
# Shipping Policy

This policy explains how Harbor Goods ships orders from our single warehouse in Millbrook and what delivery promises we make to customers. It applies to all orders placed on the Harbor Goods website from 15 January 2026. Rates for each zone are set out in the Shipping Rates and Zones document, and the choice of carrier is explained in the Carrier Rules.

## Where we ship

Harbor Goods ships to street addresses and post office boxes in all 50 US states. The continental United States is divided into Zones 1 to 4 by distance from the warehouse, and Alaska and Hawaii form Zone 5. Orders to post office boxes always travel by standard service, because express service cannot deliver to a post office box.

## Free shipping

Orders of 75 USD or more ship free by standard service to Zones 1 to 4. The threshold is measured after discounts and before tax. Orders to Zone 5 never qualify for free shipping, and the remote-area surcharge still applies to orders that do qualify. Express service is never free.

## Handling time

Our handling time is 1 business day for in-stock items. Orders placed before 1:00 PM Eastern time on a business day usually leave the warehouse the same day, and orders placed after the cut-off leave on the next business day. Business days are Monday to Friday, excluding the warehouse closure days listed in the Holiday Schedule. During the peak season set out in the Holiday Schedule, handling time may extend to 2 business days.

## Delivery times

Standard delivery takes 3 to 5 business days after the parcel leaves the warehouse. Express delivery takes 1 to 2 business days after the parcel leaves the warehouse. Zone 5 standard parcels can take up to 8 business days. Delivery times are estimates and not guarantees, but if an express parcel arrives late through a fault of ours, the customer may ask for the express part of the shipping charge back.

## Split shipments and backorders

When part of an order is on backorder, we ship the in-stock items first and send the rest when it arrives, at no extra shipping cost. Customers can cancel a backordered item at any time before it ships for a full refund of that item.

## Tracking and lost parcels

Every parcel gets a tracking number by email when it leaves the warehouse. If tracking shows no movement for 7 calendar days, customers should contact support, and we will open a trace with the carrier. If the carrier cannot find the parcel within 5 business days of the trace, we send a replacement or a full refund, whichever the customer prefers.

## Address changes

Customers can change the delivery address until the order is packed. After that, the parcel can only be redirected through the carrier, and any redirection fee charged by the carrier is passed on to the customer.
""",
)

doc(
    "shipping-rates-zones",
    "Shipping Rates and Zones",
    "2026-01",
    "2026-01-15",
    """
# Shipping Rates and Zones

This document lists the shipping charges Harbor Goods applies at checkout. All rates are in US dollars per parcel and include fuel charges. The rules on when shipping is free are in the Shipping Policy; this document covers what a parcel costs when it is not free.

## Zones

Zones are based on the distance between the Millbrook warehouse and the delivery address. Zone 1 covers addresses within 300 miles of the warehouse. Zone 2 covers 301 to 700 miles. Zone 3 covers 701 to 1,200 miles. Zone 4 covers the rest of the continental United States. Zone 5 covers Alaska and Hawaii. The zone is worked out automatically from the postal code at checkout.

## Standard rates

Standard rates apply to parcels weighing up to 5 lb.

- Zone 1 parcels cost 5.90 USD each.
- Zone 2 parcels cost 8.40 USD each.
- Zone 3 parcels cost 10.75 USD each.
- Zone 4 parcels cost 12.95 USD each.
- Zone 5 parcels cost 24.50 USD each.

Each additional pound over 5 lb adds 0.85 USD to the standard rate, in every zone. Weight is rounded up to the next full pound, so a parcel of 6.2 lb is charged as 7 lb.

## Express rates

The express rate is twice the standard rate for the same zone and weight, plus 4.00 USD. For example, a 3 lb express parcel to Zone 2 costs 20.80 USD. Express service is not available to post office boxes and is not available for parcels heavier than 30 lb.

## Remote-area surcharge

Some postal codes are hard for carriers to reach, such as island communities and mountain areas without regular road service. These codes are on the remote-area list, which the warehouse team reviews every January. A remote-area surcharge of 6.50 USD per parcel applies to every parcel sent to a postal code on the remote-area list, even when the order qualifies for free shipping. The surcharge is shown at checkout as a separate line.

## Oversize items

An oversize fee of 18.00 USD applies to any parcel longer than 48 inches on its longest side, such as boxed kayak paddles or large tent poles. The oversize fee is added on top of the standard or express rate and is never waived by a promotion.

## Multi-parcel orders

When an order ships in more than one parcel because of weight or size, each parcel is charged at its own rate. If the order qualifies for free shipping, all parcels of the order ship free except for any remote-area surcharge or oversize fee.

## Rate changes

Rates are reviewed twice a year, in January and July. Any change is published on the website at least 14 days before it takes effect. Orders placed before the change keep the rate shown at checkout.
""",
)

doc(
    "carrier-rules",
    "Carrier Rules",
    "2026-01",
    "2026-01-15",
    """
# Carrier Rules

Harbor Goods uses two carriers: NorthRoute Parcel for small parcels and Swiftline Freight for heavy shipments. This document tells the warehouse team which carrier to book for each shipment and the times by which shipments must be ready.

## Choosing a carrier

- NorthRoute Parcel carries single parcels weighing up to 30 lb.
- Swiftline Freight carries shipments weighing from 30.1 lb to 150 lb, either as one heavy carton or as several cartons strapped together.
- All express parcels go by NorthRoute Parcel, because Swiftline Freight offers ground service only.
- Shipments heavier than 150 lb need a freight quote and approval from the Operations Manager before booking.

Swiftline Freight does not deliver to Zone 5. Heavy orders for Alaska and Hawaii are split into cartons of 30 lb or less and sent by NorthRoute Parcel.

## Cut-off times

Orders released to the warehouse before 1:00 PM Eastern time are packed the same business day. NorthRoute Parcel collects from the dock at 4:00 PM Eastern time every business day. Swiftline Freight collects at 2:30 PM Eastern time, so heavy shipments must be packed and labelled by 2:00 PM to leave the same day. If a Swiftline shipment misses the 2:00 PM deadline, it is held for the next collection, and the customer is told only if the delay passes 1 business day.

## Signatures and declared value

Every shipment with an order value over 500 USD is sent with signature required on delivery. The declared value on the label is the order value before shipping, up to a maximum declared value of 2,000 USD per parcel with NorthRoute Parcel and 5,000 USD per shipment with Swiftline Freight. Higher-value orders are split so that no parcel goes over the limit.

## Dangerous goods

Fuel canisters and other items with a limited quantity label must travel by ground service. They can go by NorthRoute Parcel standard service or by Swiftline Freight, but never by express. If an order mixes fuel canisters with items the customer wants by express, the fuel canisters ship separately by ground.

## Labels and manifests

Labels are printed from the shipping system only, never handwritten. At the end of each day the shift lead closes the manifest for each carrier, and the driver signs the manifest on collection. A shipment that is not on a signed manifest is treated as still in the warehouse.

## Carrier problems

If a carrier misses a collection, the shift lead logs it and calls the carrier account desk before 5:00 PM. After two missed collections by the same carrier in one calendar month, the Operations Manager reviews whether to move volume to the other carrier for the following month.
""",
)

doc(
    "returns-policy",
    "Returns Policy",
    "2026-01",
    "2026-01-15",
    """
# Returns Policy

This is the current returns policy of Harbor Goods. It applies to all orders delivered on or after 15 January 2026 and replaces the earlier version 2025-03. Refund timing is explained in the Refunds and Chargebacks document.

## Return window

Customers may return unused items within 30 days of delivery. Unused means the item is in its original condition with all tags, parts and packaging, and has not been taken outdoors. The 30 days start on the delivery date shown in tracking. Items delivered during the peak season may have a longer window, as set out in the Holiday Schedule.

## How to start a return

Returns must be requested from the customer account or through support before anything is sent back. Each approved return gets a return authorization number that starts with RMA, for example RMA-26-10452. Parcels that arrive without a return authorization number may be refused and sent back to the customer at the customer's cost.

## Who pays for return shipping

Harbor Goods pays return shipping when an item is defective, damaged in transit or not what the customer ordered. For change-of-mind returns, a flat return label fee of 6.95 USD is deducted from the refund. Customers may use their own carrier instead, but then they pay that carrier directly and carry the risk of loss until the parcel reaches the warehouse.

## Exchanges

Each order includes one free exchange for a different size or color of the same item. The exchange item ships by standard service at no charge. A second exchange on the same order is treated as a return and a new order.

## Items that cannot be returned

- Clearance items marked final sale.
- Fuel canisters, whether opened or not.
- Personalized items, such as embroidered packs.
- Gift cards.

## Restocking fee

Opened electronics, such as GPS units and satellite messengers, carry a 15 percent restocking fee unless they are defective. No other items carry a restocking fee.

## Used items

Items that have been used outdoors cannot be returned under this policy. If a used item has a defect, the customer can make a claim under the Warranty instead.

## Refund method

Refunds go back to the original payment method. Customers who prefer store credit can choose it when they request the return, and store credit is issued as a Harbor Goods gift card.
""",
)

doc(
    "returns-policy-2025",
    "Returns Policy 2025 (superseded)",
    "2025-03",
    "2025-03-01",
    """
# Returns Policy 2025 (superseded)

Superseded on 2026-01-15 by the Returns Policy version 2026-01. This version is kept for reference only. Do not use it to answer customers about orders delivered on or after 15 January 2026.

This was the returns policy of Harbor Goods for orders delivered from 1 March 2025 until 14 January 2026.

## Return window

Customers may return unused items within 45 days of delivery. Unused meant the item was in its original condition with tags attached. The 45 days started on the date the order was delivered, and there was no extended window for holiday orders.

## How to start a return

Customers emailed support with the order number and the reason for the return. Support replied with a return form, which had to be printed and placed inside the parcel. This version did not use return authorization numbers.

## Who pays for return shipping

Customers paid the full cost of return shipping for all returns except items that arrived damaged. Harbor Goods did not provide prepaid return labels for change-of-mind returns, and customers chose their own carrier.

## Exchanges

Exchanges were not offered. Customers who wanted a different size or color had to return the item and place a new order, paying shipping on both.

## Items that cannot be returned

- Clearance items marked final sale.
- Opened fuel canisters.
- Gift cards.

## Restocking fee

All opened items carried a 20 percent restocking fee, whether or not they were electronics. Defective items were exempt from the fee.

## Used items

Used items could be returned within 14 days of delivery if they were clean and undamaged, with a 25 percent deduction from the refund. This rule was removed because many used items could not be resold.

## Store credit bonus

Customers who chose store credit instead of a refund received a 10 percent bonus on the credit amount. The bonus was removed in the 2026-01 version.

## Refund timing

Refunds were issued within 14 business days after the return arrived at the warehouse, to the original payment method or as store credit.
""",
)

doc(
    "refunds-chargebacks",
    "Refunds and Chargebacks",
    "2026-01",
    "2026-01-15",
    """
# Refunds and Chargebacks

This document explains when customers receive their money back and how the finance team handles card disputes, also called chargebacks.

## Inspection and refund timing

Returned items are inspected within 3 business days of arriving at the warehouse. Refunds are issued within 5 business days after the returned item passes inspection. The customer receives an email when the refund is issued.

Card refunds can take up to 10 business days to appear on the customer's statement, depending on the bank. Store credit refunds appear in the customer account within 1 hour of being issued.

## Partial refunds

If a returned item is missing parts or tags, the finance team may deduct up to 50 percent of the item price from the refund. The deduction and the reason are stated in the refund email. Shipping charges from the original order are refunded only when the return is due to a defect, transit damage or our error.

## Refunds for cancelled orders

Orders cancelled before they are packed are refunded in full within 2 business days. Orders cancelled after packing are handled as returns.

## Chargebacks

A chargeback happens when a customer asks their bank to reverse a payment. When the payment processor tells us about a chargeback, the finance team must respond within 7 calendar days with evidence.

Evidence for a chargeback includes:

- the carrier tracking record showing delivery to the order address,
- the signature on delivery for orders over 500 USD,
- the order confirmation email and the IP address used to place the order,
- all messages between the customer and support about the order.

Chargebacks for amounts under 25 USD are accepted without a contest, because the cost of responding is higher than the amount. The finance team still records them to spot repeat patterns.

## Chargebacks during a return

If a customer files a chargeback while a return is in progress, the refund is paused until the bank decides the dispute. This prevents the customer from being paid twice. Support tells the customer about the pause in writing.

## Repeat disputes

A customer account with 3 chargebacks in 12 months is reviewed by the finance team, and the account may be closed after the review. The customer is told the outcome by email.

## Fees

The payment processor charges Harbor Goods a fee of 15 USD for every chargeback, whether we win or lose. This fee is never passed on to the customer.
""",
)

doc(
    "warranty",
    "Warranty",
    "2026-01",
    "2026-01-15",
    """
# Warranty

This document sets out the warranty Harbor Goods gives on the products it sells. The warranty covers defects in materials and workmanship that appear under normal use. It is in addition to the customer's rights under the Returns Policy and does not replace them.

## Coverage periods

Coverage starts on the delivery date shown in tracking and depends on the product class. Tents and shelters are covered for 2 years from the date of delivery. Sleeping bags and sleeping pads are covered for 2 years from the date of delivery. Harbor Line backpacks, our own brand, are covered for 5 years from the date of delivery. Backpacks from other brands are covered for 1 year. Footwear is covered for 1 year from the date of delivery. Electronics, such as headlamps, GPS units and satellite messengers, are covered for 1 year from the date of delivery. Cooking stoves are covered for 1 year, but stove fuel and other consumables are not covered.

The warranty belongs to the original buyer and cannot be transferred. A replacement product is covered for the rest of the original coverage period or for 90 days, whichever is longer.

## Who gives the warranty

Harbor Goods handles every warranty claim directly, even when the product was made by a supplier. Customers never need to contact the maker. Harbor Goods then recovers the cost from the supplier under the supplier terms.

<!-- pagebreak -->

## Exclusions

The warranty does not cover normal wear and tear, such as faded colors, worn soles, or abrasion of fabrics after heavy use. It does not cover damage caused by misuse, accidents, fire, or improper storage, such as packing a tent away wet for long periods. UV damage to tent fabrics is excluded after the first 12 months. Products used for commercial rental or for group instruction are covered for 90 days only. Products bought from sellers other than Harbor Goods are not covered. Any modification, such as cutting, re-sealing seams with products we did not approve, or replacing parts with parts we did not supply, ends the warranty for that product.

Zippers, buckles and other small hardware are covered for the full period of the product, and we usually fix them by repair rather than replacement.

## Repair, replacement or refund

When a claim is approved, Harbor Goods first tries to repair the product. If the repair would cost more than 60 percent of the price paid, we replace the product with the same item or the closest current model. If neither repair nor replacement is possible, we refund the price paid. We never refund shipping charges from the original order under the warranty.

<!-- pagebreak -->

## How to make a claim

Customers must make a claim within 30 days of discovering the defect and before the coverage period ends. Claims are made with warranty claim form WC-1, which is available from the customer account or from support. The claim must include proof of purchase and at least two photos showing the defect.

Harbor Goods decides each claim within 10 business days of receiving the complete claim form. If we need to see the product, we email a prepaid return label, and Harbor Goods pays shipping both ways within the United States. Repaired or replacement items are sent by standard service.

## Claim records

Each claim gets a number that starts with WAR, and the decision is sent by email. If a claim is refused, the email gives the reason, and the customer may ask for a second review by a team lead within 14 days of the decision. The second review is final.
""",
    pdf=True,
)

doc(
    "supplier-terms-alpine",
    "Supplier Terms - Alpinewood Supply Co.",
    "2026-01",
    "2026-01-01",
    """
# Supplier Terms - Alpinewood Supply Co.

This document records the trading terms agreed between Harbor Goods and Alpinewood Supply Co., referred to here as Alpinewood. Alpinewood supplies tents, shelters, sleeping bags and sleeping pads. These terms apply to every purchase order Harbor Goods sends to Alpinewood from 1 January 2026 until 31 December 2027, unless both parties agree a change in writing.

## Ordering

Every order must be placed with a Harbor Goods purchase order. The minimum order value is 2,500 USD per purchase order, measured before freight. Orders below the minimum are accepted only if Harbor Goods pays a small-order fee of 75 USD. Alpinewood confirms each purchase order in writing within 2 business days and states the expected ship date.

## Prices

Prices are fixed for each calendar year and listed in the Alpinewood price list. Alpinewood must give 60 days written notice of any price change, and a change never applies to purchase orders already confirmed.

## Payment terms

Payment is due net 30 from the invoice date. Harbor Goods may take a 2 percent early payment discount if it pays the full invoice within 10 days of the invoice date. Invoices must show the Harbor Goods purchase order number and must be sent only after the goods have shipped. Payment is made by bank transfer in US dollars.

<!-- pagebreak -->

## Delivery

The standard lead time is 21 calendar days from confirmation of the purchase order to delivery at the Harbor Goods warehouse. Goods are delivered to the warehouse dock on weekdays during receiving hours. Freight is prepaid by Alpinewood on orders of 5,000 USD or more; on smaller orders, freight is added to the invoice at cost.

## Late delivery

If a delivery arrives later than the confirmed date, Alpinewood credits Harbor Goods 1 percent of the purchase order value for each full week of delay, capped at 5 percent of the purchase order value. Harbor Goods may cancel any part of an order that is more than 4 weeks late without paying for it.

## Packaging and labels

Each carton must carry a label with the Harbor Goods SKU, such as HG-TENT-220, the quantity and the purchase order number. Cartons must not weigh more than 50 lb.

<!-- pagebreak -->

## Defects

Harbor Goods reports manufacturing defects to Alpinewood within 14 days of receipt for defects found at receiving, and at any time during the warranty period for defects found later by customers. Alpinewood replaces or credits defective goods, at its own choice, and collects them at its own cost.

## Warranty support

Alpinewood reimburses Harbor Goods for warranty claims on Alpinewood products that Harbor Goods approves under its own warranty, for up to 2 years from delivery to the customer.

## Ending the agreement

Either party may end these terms with 90 days written notice. Purchase orders confirmed before the notice date must still be delivered and paid for.
""",
    pdf=True,
)

doc(
    "supplier-terms-trailworks",
    "Supplier Terms - Trailworks Equipment Co.",
    "2026-02",
    "2026-02-01",
    """
# Supplier Terms - Trailworks Equipment Co.

This document records the trading terms agreed between Harbor Goods and Trailworks Equipment Co., referred to here as Trailworks. Trailworks supplies backpacks, including the Harbor Line backpacks made for our own brand, as well as trekking poles, cooking stoves and headlamps. These terms apply to every purchase order Harbor Goods sends to Trailworks from 1 February 2026 until 31 January 2028.

## Ordering

Every order must be placed with a Harbor Goods purchase order. The minimum order value is 4,000 USD per purchase order, measured before freight. Trailworks does not accept orders below the minimum. Harbor Line backpacks must be ordered in multiples of 24 units per color. Trailworks confirms each purchase order in writing within 3 business days.

## Prices

Prices are reviewed every six months, in January and July. Trailworks must give 90 days written notice of any price change, and a change never applies to purchase orders already confirmed.

## Payment terms

Payment is due net 45 from the invoice date. Trailworks does not offer an early payment discount. Invoices must show the Harbor Goods purchase order number and are issued on the day the goods leave the Trailworks factory. Payment is made by bank transfer in US dollars.

<!-- pagebreak -->

## Delivery

The standard lead time is 35 calendar days from confirmation of the purchase order to delivery at the Harbor Goods warehouse. For Harbor Line backpacks, the lead time is 50 calendar days. Harbor Goods pays the freight on all orders, unless an order is worth 10,000 USD or more, in which case Trailworks pays the freight.

## Late delivery

If a delivery arrives later than the confirmed date, Trailworks credits Harbor Goods 2 percent of the purchase order value for each full week of delay, capped at 10 percent of the purchase order value. Harbor Goods may cancel any part of an order that is more than 6 weeks late without paying for it.

## Packaging and labels

Each carton must carry a label with the Harbor Goods SKU, such as HG-PACK-450, the quantity and the purchase order number. Cartons must not weigh more than 40 lb.

<!-- pagebreak -->

## Defects

Harbor Goods reports manufacturing defects to Trailworks within 30 days of receipt. Trailworks replaces defective goods in the next shipment and does not issue cash credits unless the item is discontinued. Harbor Goods holds defective goods for collection for up to 60 days.

## Warranty support

Trailworks reimburses Harbor Goods for approved warranty claims on Trailworks products for up to 1 year from delivery to the customer, and for up to 5 years for Harbor Line backpacks.

## Ending the agreement

Either party may end these terms with 120 days written notice. Purchase orders confirmed before the notice date must still be delivered and paid for.
""",
    pdf=True,
)

doc(
    "purchase-orders",
    "Purchase Orders",
    "2026-01",
    "2026-01-15",
    """
# Purchase Orders

This procedure explains how Harbor Goods staff raise and approve purchase orders, called POs, for goods and services from suppliers. No supplier may be asked to ship or start work without an approved PO, except in the emergency case described below.

## Raising a PO

POs are raised in the purchasing system by the buyer responsible for the product category. Each PO gets a number in the format PO-2026-0001, counting up through the year. The PO must list the supplier, each SKU, the quantity, the unit price, the requested delivery date and the total value in US dollars. Before raising a PO, the buyer checks that the order meets the supplier's minimum order value in the supplier terms.

## Approval limits

- Purchase orders up to 1,000 USD can be approved by a team lead.
- Purchase orders above 1,000 USD and up to 10,000 USD need approval from the Operations Manager.
- Purchase orders above 10,000 USD need approval from both the Operations Manager and the Finance Director.

The approval limit is based on the total value of the PO including freight and before any early payment discount. Nobody may approve a PO they raised themselves.

## No splitting

Splitting one purchase into several smaller POs to stay under an approval limit is not allowed. POs to the same supplier for the same SKUs within 7 calendar days are added together when checking the limit. A breach is reported to the Finance Director.

## New suppliers

Before the first PO to a new supplier, the buyer completes vendor form VF-2 and gets approval from the Finance Director. For any purchase above 5,000 USD from a new supplier, the buyer must also collect at least 2 written quotes.

## Emergency purchases

When the warehouse needs something urgently, such as packing tape on a busy day, a team lead may buy up to 500 USD on the company card without a PO. A PO must then be raised within 2 business days and marked as an emergency purchase.

## Changes and cancellations

Any change to quantity or price on an approved PO goes back for approval if it raises the total by more than 10 percent. Cancellations are sent to the supplier in writing, and the buyer records the reason in the PO.

## Closing a PO

A PO is closed when all goods are received and the invoice is paid, or after 90 days with no activity. Open POs older than 60 days are reviewed every month by the Operations Manager.
""",
)

doc(
    "invoice-rules",
    "Invoice Rules",
    "2026-01",
    "2026-01-15",
    """
# Invoice Rules

This procedure explains how the accounts payable team checks and pays supplier invoices. The goal is to pay the right amount, once, on time.

## Receiving invoices

Suppliers send invoices to the accounts payable inbox. Every invoice must show a Harbor Goods PO number. Invoices without a PO number are returned to the supplier within 3 business days with a request to add it. The team logs each invoice on the day it arrives and checks the invoice number against earlier invoices to stop duplicate payments.

## Three-way match

Before any invoice is paid, the team performs a three-way match. The invoice, the purchase order and the receiving report must agree on the items, quantities and prices. Quantities must match exactly. A price difference is accepted if it is no more than 2 percent of the line value or 50 USD, whichever is lower. Any larger difference is sent to the buyer, who must resolve it with the supplier within 5 business days.

If the receiving report shows a shortage or damage, the team pays only for the goods received in good condition and asks the supplier for a credit note for the rest.

## Payment timing

Invoices are paid according to the payment terms in each supplier's terms document, counted from the invoice date. Payment runs take place every Thursday. An invoice that falls due before the next Thursday run is paid in the current run, so that no invoice is paid late because of the run schedule. Early payment discounts are taken whenever a supplier offers them and the three-way match is complete in time.

## Late fee

Suppliers may charge a late fee of 1.5 percent per month on any invoice amount that Harbor Goods pays after the due date. The fee is worked out per full month of delay. Late fees are charged to the budget of the department that caused the delay, and the accounts payable lead reviews every late fee at the end of the month.

## Approvals

Invoices that pass the three-way match are approved by the accounts payable lead. Invoices above 25,000 USD also need sign-off from the Finance Director before payment, even if the match is clean.

## Disputes

When Harbor Goods disputes an invoice in good faith, the disputed part is not treated as late while the dispute is open. The undisputed part must still be paid on time.

## Records

Paid invoices and their matching documents are stored in the finance system for 7 years.
""",
)

doc(
    "payment-methods",
    "Payment Methods",
    "2026-01",
    "2026-01-15",
    """
# Payment Methods

This page lists the ways customers can pay for orders on the Harbor Goods website and the rules that apply to each.

## Accepted methods

- Credit and debit cards from the major card networks.
- Digital wallets on phones and computers that support card payments.
- Harbor Goods gift cards.
- Pay in 4, an installment option for orders from 100 USD to 1,500 USD.
- Bank transfer, for business and club orders over 1,000 USD only.

Harbor Goods does not accept cash on delivery, personal checks or payment by phone.

## Fees

Harbor Goods does not add a surcharge for any payment method. Pay in 4 splits the order total into four equal payments taken every two weeks, with no interest and no fee when payments are made on time. A late Pay in 4 payment carries a fee of 7 USD, charged by the installment provider and not by Harbor Goods.

## Gift cards

Gift cards are sold in values from 10 USD to 500 USD. Gift card codes start with HGC followed by 12 digits, and gift cards never expire. A gift card can be combined with one other payment method on the same order. Gift cards cannot be used to buy other gift cards, and they cannot be exchanged for cash except where the law requires it.

## Fraud review

Orders over 2,000 USD paid by card are held for a fraud review of up to 24 hours before they are released to the warehouse. The customer is told about the review by email. Orders over 300 USD where the billing name and the delivery name do not match are also reviewed.

## Bank transfer orders

Business and club customers paying by bank transfer receive a proforma invoice with the order number as the payment reference. The order ships once the payment arrives in full, which usually takes 1 to 3 business days. Approved school and club accounts that have placed at least 3 bank transfer orders may ask for net 15 terms on later orders.

## Currency and tax

All prices are in US dollars. Sales tax is calculated at checkout from the delivery address. When an order is refunded, the refund goes back to the payment method used, and gift card amounts go back to the gift card.

## Payment failures

If a card payment fails, the order stays open for 48 hours so the customer can try another method. After 48 hours, the order is cancelled and any reserved stock is released.
""",
)

doc(
    "inventory-counts",
    "Inventory Counts",
    "2026-01",
    "2026-01-15",
    """
# Inventory Counts

This standard operating procedure explains how the warehouse team counts stock so that the quantities in the inventory system match what is on the shelves.

## ABC classes

Each SKU is placed in a class once a quarter, based on its sales value over the last 12 months. Class A holds the top 20 percent of SKUs by sales value, class B the next 30 percent, and class C the remaining 50 percent. Tents such as HG-TENT-220 and high-value electronics are always class A, whatever their sales.

## Count frequency

- Class A SKUs are counted every week.
- Class B SKUs are counted every month.
- Class C SKUs are counted every quarter.

In addition, a full physical count of the whole warehouse takes place once a year on the last Sunday of January.

## Blind counts

Counts are blind. The counter scans the bin location, such as BIN A-04-2, and enters the quantity found without seeing the system quantity. This keeps counters from simply confirming the expected number.

## Variance limit

After each count, the system compares the counted quantity with the system quantity. A variance of more than 3 percent of the system quantity triggers a recount by a second counter. For SKUs with a system quantity under 20 units, any difference of 2 units or more triggers a recount.

If the recount confirms the variance, the shift lead posts an adjustment within 1 business day and records the likely cause, such as a picking error, damage or a receiving mistake. Any adjustment worth more than 500 USD at cost must be reported to the Operations Manager on the same day.

## Count timing

Cycle counts happen before 10:00 AM, before picking gets busy. Bins being counted are locked in the system, so no picks or put-aways can happen in them during the count, which normally takes under 15 minutes per bin.

## Monthly review

At the start of each month, the inventory lead reports the count accuracy for the previous month. The target is at least 98 percent of counted bins matching the system without adjustment. If accuracy falls below 95 percent for two months in a row, class B SKUs move to counting every two weeks until accuracy recovers.

## Damaged stock found during counts

Damaged items found during a count are moved to the damage bin and handled under the Damaged Goods procedure. They are not counted as available stock.
""",
)

doc(
    "receiving-goods",
    "Receiving Goods",
    "2026-01",
    "2026-01-15",
    """
# Receiving Goods

This standard operating procedure covers how the warehouse team receives deliveries from suppliers, checks them and makes them available for sale.

## Receiving hours

Supplier deliveries are accepted at dock 2 from 7:00 AM to 3:00 PM, Monday to Friday. Deliveries arriving outside these hours are turned away unless the Operations Manager has agreed a special time in advance. Suppliers must book a delivery slot at least 24 hours ahead.

## Checks at the dock

Before signing for a delivery, the receiver counts the cartons and compares the number with the bill of lading. Any shortage or visible damage to the cartons must be written on the delivery receipt before the driver leaves. A clean signature makes later claims against the carrier much harder.

## Inspection

- Match each carton label to an open purchase order and SKU, for example HG-TENT-220.
- Open at least 10 percent of the cartons in each delivery, with a minimum of 2 cartons, and check the contents against the packing list.
- Weigh any carton that seems too heavy and compare it with the carton weight limit in the supplier terms.
- Check that tents and sleeping bags are dry and do not smell of damp.

Goods waiting for inspection stay in the quarantine area and must be inspected within 2 business days of delivery.

## Damage and shortages

Visible transport damage must be reported to the carrier and the supplier within 48 hours of delivery, with photos of the damaged cartons and contents. Shortages found after opening cartons are reported to the supplier within the same 48 hours. Manufacturing defects found at receiving are reported under the time limits in each supplier's terms.

## Receiving report

The receiver enters a receiving report in the inventory system on the day of delivery, listing the quantity received in good condition for each SKU. The receiving report is the third document in the three-way match used by accounts payable, so it must be accurate. Damaged units are recorded separately and moved to the damage bin.

## Put-away

Once the receiving report is posted, goods are moved to their bin locations. Class A items are put away the same day; other items are put away within 1 business day. The put-away is confirmed by scanning the bin label.

## Returns to suppliers

Goods rejected at receiving are labelled with the supplier return number and stored in the returns cage until the supplier collects them.
""",
)

doc(
    "packing",
    "Packing",
    "2026-01",
    "2026-01-15",
    """
# Packing

This standard operating procedure explains how orders are packed at the Harbor Goods pack stations so that they arrive safely and at the lowest shipping cost.

## Before packing

The packer scans the order barcode and then scans each item. The system warns if an item is missing or wrong. Every order gets a packing slip, and gift orders get a gift slip with no prices shown.

## Choosing a box

Use the smallest box that leaves at least 2 inches of padding between the items and every side of the box. The pack station screen suggests a box size from the item dimensions; packers may choose a larger box only if the items do not fit. Boxes are numbered by size from B1, the smallest, to B9, the largest. Kayak paddles and tent poles longer than 48 inches go in the long tube box, code T1.

A single carton must not weigh more than 30 lb, so that it can travel with NorthRoute Parcel. Orders heavier than that are split into several cartons or sent by Swiftline Freight as set out in the Carrier Rules.

## Filling and sealing

Fill empty space with recycled paper fill. Foam peanuts are not used. Seal each box with an H-pattern of tape across the top and bottom seams.

## Fragile items

Fragile items, such as glass lanterns, ceramic mugs and solar panels, are wrapped in two layers of bubble wrap and placed in the center of the box with paper fill on all sides. Fragile boxes get a FRAGILE label on two opposite sides and on the top. Do not pack fragile items in the same box as heavy items such as cast iron cookware.

## Fuel canisters

Fuel canisters are packed on their own in a box with a limited quantity label and are always sent by ground service. They are never packed in the same box as other items and never sent by express.

## Weighing and labelling

Every packed box is weighed on the station scale. If the weight differs from the system estimate by more than 10 percent, the packer opens the box and checks the contents again. The shipping label is then printed and placed on the largest flat side, away from seams.

## Quality checks

The shift lead checks 1 in every 50 packed orders at random before it is sealed. Any packing error is recorded against the packer, and a packer with 3 errors in a week gets a refresher session.

## Pack targets

The standard pack rate is 25 single-item orders per hour per packer. During peak season the target stays the same, and extra staff are added instead.
""",
)

doc(
    "damaged-goods",
    "Damaged Goods",
    "2026-01",
    "2026-01-15",
    """
# Damaged Goods

This standard operating procedure explains how Harbor Goods handles items that arrive damaged at a customer's address and items found damaged inside the warehouse.

## Customer reports

Customers must report transit damage within 7 days of delivery. The report must include at least 3 photos: one of the outer box, one of the packing inside, and one of the damaged item. Support creates a damage case with a number that starts with DMG.

## Replacement or refund

If the same item is in stock, Harbor Goods ships a replacement by express service at no charge, and the customer does not wait for the damaged item to come back. If the item is out of stock, the customer receives a full refund, including the original shipping charge. Customers may choose a refund instead of a replacement in every case.

## Returning the damaged item

When the damaged item is worth less than 40 USD, the customer does not need to return it and may keep or dispose of it. For items worth 40 USD or more, support emails a prepaid return label, and the customer has 14 days to send the item back. If the item is not sent back within 14 days, a team lead reviews the case.

## Carrier claims

Support files a claim with the carrier within 5 business days of the customer's report, attaching the customer's photos and the packing record from the pack station. Claim payments from carriers go to the finance team and do not change what the customer receives.

## Damage found in the warehouse

Items found damaged in the warehouse are moved to the damage bin DMG-01 and recorded in the inventory system on the same day. Each week the inventory lead decides for every item whether to repair it, sell it as B-stock, return it to the supplier or dispose of it.

- Items with only cosmetic marks are sold as B-stock at 30 percent off the full price.
- Items with a manufacturing defect go back to the supplier under the supplier terms.
- Items that are unsafe, such as stoves with bent valves or dented fuel canisters, are disposed of and never sold.

## Repeat damage

If the same SKU is reported damaged in transit 3 times in 30 days, the packing lead reviews its packing method and may move it to a larger box or add extra padding.

## Records

Every damage case records the order number, the SKU, the photos, the decision and the cost. The Operations Manager reviews damage costs every month.
""",
)

doc(
    "support-hours-sla",
    "Support Hours and Service Levels",
    "2026-01",
    "2026-01-15",
    """
# Support Hours and Service Levels

This document sets out when the Harbor Goods support team is available and how quickly customers can expect an answer.

## Support hours

- Monday to Friday, 8:00 AM to 6:00 PM Eastern time.
- Saturday, 9:00 AM to 1:00 PM Eastern time.
- Closed on Sundays and on the closure days listed in the Holiday Schedule.

Customers can reach support by email, by live chat on the website and by phone. Live chat and phone are only staffed during support hours. Emails can be sent at any time and are answered in order of arrival, except that priority tickets go first.

## First-response targets

The first-response target for email is 4 business hours. Live chats are answered within 2 minutes during support hours. Phone calls are answered within 3 minutes on average. Business hours means the support hours above, so an email sent on Saturday evening falls due on Monday morning.

## Priority tickets

Tickets about a lost parcel, a damaged order, a payment taken twice or a safety concern with a product are priority tickets. Priority tickets get a first response within 2 business hours. Safety concerns, such as a stove that leaks fuel, are also passed to the Operations Manager on the same day.

## Resolution targets

The team aims to resolve 90 percent of tickets within 3 business days of the first message. Tickets that depend on a carrier trace or a warranty decision may take longer, and the customer is updated at least every 2 business days until the case is closed.

## Escalation

A ticket is escalated to a team lead when the customer has replied twice without the problem being solved, or when the customer asks for a manager. Team leads reply to escalated tickets within 1 business day.

## Language and tone

Support replies are written in plain English. Agents give their first name and never promise anything the policies do not allow. If a request falls outside policy, the agent explains why and offers what the policy does allow.

## Surveys and quality

A short satisfaction survey is sent 24 hours after a ticket is closed. The target satisfaction score is at least 4.5 out of 5. Team leads review 5 closed tickets per agent every week.

## Out-of-hours messages

Outside support hours, live chat shows a message with the next opening time and offers to turn the chat into an email ticket.
""",
)

doc(
    "data-retention",
    "Data Retention and Privacy",
    "2025-12",
    "2025-12-01",
    """
# Data Retention and Privacy

This document explains what personal data Harbor Goods keeps, how long we keep it, and how customers can ask us to delete it. It covers customer data and the contact details of supplier staff held in our systems.

## What we collect

We collect the data needed to take and deliver orders: name, delivery and billing address, email address, phone number, order history and messages with support. Harbor Goods never stores full payment card numbers. Card details are handled by our payment processor, and we keep only the last four digits and the card type for each order.

## Retention periods

Order records, including invoices and payment records, are kept for 7 years after the end of the financial year in which the order was placed, to meet tax and accounting rules. Customer account data is kept while the account is active and is deleted after 24 months without a login or an order. Support tickets and chat transcripts are kept for 3 years after the ticket is closed. Marketing consent records are kept for 5 years after the customer withdraws consent, so that we can prove what was agreed. Warehouse security camera footage is kept for 30 days unless it is needed for an investigation.

<!-- pagebreak -->

## Deletion requests

Customers can ask Harbor Goods to delete their personal data by using the privacy request form in their account or by writing to support. We verify the identity of the person making the request within 5 business days, usually by a code sent to the email address on the account. Once the identity is verified, we complete the deletion within 30 days and send a confirmation email.

Some data cannot be deleted on request. Order records that we must keep for tax purposes are kept for the rest of their 7 year period, but the name and contact details on them are replaced with a reference number where the law allows. Data needed for an open chargeback, warranty claim or legal claim is kept until that matter is closed.

## Marketing emails

Customers can unsubscribe from marketing emails at any time with the unsubscribe option at the bottom of each email. Unsubscribe requests take effect within 2 business days. Order and shipping emails are still sent because they are needed to deliver the order.

<!-- pagebreak -->

## Backups

Backups of the order and customer systems are kept for 35 days. Data deleted from the live systems disappears from backups when the backups that hold it expire, so a deletion is complete in every system no later than 35 days after it is made in the live system.

## Access and security

Staff can see customer data only when their role needs it. Access rights are reviewed every quarter by the Operations Manager. If a data breach affects customer data, the incident lead informs management within 24 hours of discovery, and affected customers are told without undue delay.

## Data sharing

We share delivery details with NorthRoute Parcel and Swiftline Freight to deliver orders, and payment details with our payment processor. Harbor Goods does not sell customer data to anyone.
""",
    pdf=True,
)

doc(
    "holiday-schedule",
    "Holiday Schedule 2026",
    "2026-01",
    "2026-01-01",
    """
# Holiday Schedule 2026

This document lists the days in 2026 when the Harbor Goods warehouse and support team are closed, the peak season rules and the last order days for delivery before the winter holidays.

## Closure days

The warehouse and support team are closed on the following days in 2026:

- Thursday 1 January, New Year's Day.
- Monday 25 May, Memorial Day.
- Friday 3 July, observed for Independence Day.
- Monday 7 September, Labor Day.
- Thursday 26 November, Thanksgiving Day.
- Friday 25 December, Christmas Day.

On Thursday 24 December and Thursday 31 December, the warehouse closes at 12:00 PM and the order cut-off for same-day shipping moves to 10:00 AM Eastern time. Support is open from 8:00 AM to 12:00 PM on those days. Orders placed on closure days are processed on the next business day.

## Peak season

Peak season runs from Friday 27 November to Thursday 24 December 2026. During peak season, handling time may extend to 2 business days, because order volume is several times higher than usual. The warehouse runs a Saturday shift during peak season to keep up, but Saturdays still do not count as business days for handling time.

## Last order days

To receive an order before Christmas Day, customers should order by these dates:

- The last order day for standard delivery is Wednesday 16 December 2026.
- The last order day for express delivery is Monday 21 December 2026, before 1:00 PM Eastern time.
- Zone 5 orders for Alaska and Hawaii should be placed by Friday 11 December 2026 for standard delivery.

Orders placed after these dates are shipped as normal but may arrive after Christmas Day.

## Extended returns

Items delivered between 15 November and 24 December 2026 can be returned until 31 January 2027, if they meet the other conditions of the Returns Policy. This extension replaces the usual 30-day window only where it gives the customer more time.

## Staff leave

Warehouse staff may not take annual leave during peak season, except with approval from the Operations Manager requested before 1 October. Support staff follow the same rule, so that live chat and phone hours stay the same throughout the season.
""",
)

doc(
    "promo-price-rules",
    "Promotions and Price Rules",
    "2026-01",
    "2026-01-15",
    """
# Promotions and Price Rules

This document sets the rules for discount codes, sales and price changes at Harbor Goods. It applies to everyone who creates promotions or edits prices in the online shop.

## Promo codes

- Only one promo code can be used per order.
- Promo codes cannot be combined with each other or with the employee discount.
- Promo codes can be combined with free shipping, but the free-shipping threshold is checked after the code is applied.
- Promo codes do not apply to gift cards or to clearance items marked final sale.

Promo codes are written in capital letters, such as TRAIL15 for 15 percent off. Each code has an end date and stops working at 11:59 PM Eastern time on that date. Codes for single customers, such as service recovery codes from support, are limited to one use and a maximum value of 25 USD.

## Approval of promotions

Every promotion needs a written brief with the discount, the products, the start and end dates and the expected cost. Promotions offering up to 25 percent off can be approved by the Marketing Lead. Any promotion offering more than 25 percent off needs written approval from the Finance Director as well. No promotion may bring the selling price of an item below its cost plus 10 percent.

## Price changes

Regular price changes are made on Mondays only, so that the warehouse and support teams know what to expect. A price change affecting more than 50 SKUs at once needs approval from the Operations Manager. Every price change is recorded with the old price, the new price, the date and the person who made it.

## Price adjustment after purchase

If Harbor Goods lowers the price of an item within 14 days after a customer's order, the customer can ask support for the difference once per order. The difference is refunded to the original payment method. Price adjustments do not apply to flash sales lasting less than 24 hours or to clearance items.

## Employee discount

Employees receive 30 percent off full-price items for personal use, up to 1,500 USD of purchases per calendar year. The employee discount cannot be combined with any promo code or sale price.

## Clearance

Items move to clearance when they have not sold for 180 days or when a newer model replaces them. Clearance items are marked final sale and cannot be returned, as stated in the Returns Policy. Clearance prices never go below the cost of the item.
""",
)


# ---------------------------------------------------------------------------------------------
# Questions (answer keys). Key order in each object is fixed by q() below.
# For PDF docs, the page is the 1-based page section of the source; markdown docs use None.

QUESTIONS: list[dict] = []


def q(
    qtype: str,
    question: str,
    doc_ids: list[str],
    quotes: list[str],
    pages: list[int | None],
    must_contain: list[str],
    tags: list[str] | None = None,
    must_not_appear: list[str] | None = None,
    notes: str = "",
) -> None:
    QUESTIONS.append(
        {
            "id": f"Q{len(QUESTIONS) + 1:03d}",
            "type": qtype,
            "question": question,
            "expected_doc_ids": doc_ids,
            "expected_quotes": quotes,
            "expected_pages": pages,
            "must_contain": must_contain,
            "tags": tags or [],
            "must_not_appear": must_not_appear or [],
            "notes": notes,
        }
    )


def single(question, doc_id, quotes, pages, must_contain, tags=None, notes=""):
    q("single", question, [doc_id], quotes, pages, must_contain, tags, None, notes)


def multi(question, doc_ids, quotes, pages, must_contain, tags=None, notes=""):
    q("multi", question, doc_ids, quotes, pages, must_contain, tags, None, notes)


def unanswerable(question, must_not_appear, notes=""):
    q("unanswerable", question, [], [], [], [], None, must_not_appear, notes)


EXACT = ["exact-term"]
PARA = ["paraphrase"]

# --- single (40) -----------------------------------------------------------------------------

single(
    "How many days do customers have to return unused items?",
    "returns-policy",
    ["Customers may return unused items within 30 days of delivery."],
    [None],
    ["30"],
    notes="decoy: returns-policy-2025 says 45 days",
)
single(
    "Who covers the postage when I send back something I simply changed my mind about?",
    "returns-policy",
    [
        "For change-of-mind returns, a flat return label fee of 6.95 USD is deducted from the refund."
    ],
    [None],
    ["6.95"],
    PARA,
    "decoy: returns-policy-2025 made the customer pay the full return shipping cost",
)
single(
    "What restocking fee applies to opened electronics?",
    "returns-policy",
    [
        "Opened electronics, such as GPS units and satellite messengers, carry a 15 percent restocking fee unless they are defective."
    ],
    [None],
    ["15 percent"],
    EXACT,
    "decoy: returns-policy-2025 charged 20 percent on all opened items",
)
single(
    "Can a customer exchange an item for a different size?",
    "returns-policy",
    ["Each order includes one free exchange for a different size or color of the same item."],
    [None],
    ["one free exchange"],
    notes="decoy: returns-policy-2025 offered no exchanges",
)
single(
    "When was the 2025 returns policy superseded, and by which version?",
    "returns-policy-2025",
    ["Superseded on 2026-01-15 by the Returns Policy version 2026-01."],
    [None],
    ["2026-01-15", "2026-01"],
    notes="asks about the old decoy document itself",
)
single(
    "What are the payment terms with Alpinewood?",
    "supplier-terms-alpine",
    ["Payment is due net 30 from the invoice date."],
    [1],
    ["net 30"],
    EXACT,
    "decoy: Trailworks is net 45 with the same sentence shape; page 1",
)
single(
    "What are the payment terms with Trailworks?",
    "supplier-terms-trailworks",
    ["Payment is due net 45 from the invoice date."],
    [1],
    ["net 45"],
    EXACT,
    "decoy: Alpinewood is net 30; page 1",
)
single(
    "What is the minimum order value for a Trailworks purchase order?",
    "supplier-terms-trailworks",
    ["The minimum order value is 4,000 USD per purchase order, measured before freight."],
    [1],
    ["4,000"],
    EXACT,
    "decoy: Alpinewood minimum is 2,500 USD; page 1",
)
single(
    "What does Alpinewood owe Harbor Goods when a delivery is late?",
    "supplier-terms-alpine",
    [
        "Alpinewood credits Harbor Goods 1 percent of the purchase order value for each full week of delay, capped at 5 percent of the purchase order value."
    ],
    [2],
    ["1 percent", "5 percent"],
    notes="decoy: Trailworks credits 2 percent per week capped at 10 percent; page 2",
)
single(
    "What is the standard lead time for Trailworks orders?",
    "supplier-terms-trailworks",
    [
        "The standard lead time is 35 calendar days from confirmation of the purchase order to delivery at the Harbor Goods warehouse."
    ],
    [2],
    ["35"],
    notes="decoy: Alpinewood lead time is 21 calendar days; page 2",
)
single(
    "Do we get anything off if we settle an Alpinewood bill quickly?",
    "supplier-terms-alpine",
    [
        "Harbor Goods may take a 2 percent early payment discount if it pays the full invoice within 10 days of the invoice date."
    ],
    [1],
    ["2 percent", "10 days"],
    PARA,
    "decoy: Trailworks offers no early payment discount; page 1",
)
single(
    "How long does Harbor Goods have to report manufacturing defects to Trailworks?",
    "supplier-terms-trailworks",
    ["Harbor Goods reports manufacturing defects to Trailworks within 30 days of receipt."],
    [3],
    ["30 days"],
    notes="decoy: Alpinewood allows 14 days for defects found at receiving; page 3",
)
single(
    "How long are tents covered by the warranty?",
    "warranty",
    ["Tents and shelters are covered for 2 years from the date of delivery."],
    [1],
    ["2 years"],
    notes="page 1",
)
single(
    "How long is the warranty on Harbor Line backpacks?",
    "warranty",
    ["Harbor Line backpacks, our own brand, are covered for 5 years from the date of delivery."],
    [1],
    ["5 years"],
    notes="page 1; other backpack brands get 1 year",
)
single(
    "My tent fabric was damaged by the sun after two seasons of use. Will the warranty pay for it?",
    "warranty",
    ["UV damage to tent fabrics is excluded after the first 12 months."],
    [2],
    ["12 months"],
    PARA,
    "page 2",
)
single(
    "What happens to a warranty claim when the repair would cost too much?",
    "warranty",
    [
        "If the repair would cost more than 60 percent of the price paid, we replace the product with the same item or the closest current model."
    ],
    [2],
    ["60 percent"],
    notes="page 2",
)
single(
    "Which form is used to make a warranty claim?",
    "warranty",
    [
        "Claims are made with warranty claim form WC-1, which is available from the customer account or from support."
    ],
    [3],
    ["WC-1"],
    EXACT,
    "page 3",
)
single(
    "How long does Harbor Goods keep order records?",
    "data-retention",
    [
        "Order records, including invoices and payment records, are kept for 7 years after the end of the financial year in which the order was placed, to meet tax and accounting rules."
    ],
    [1],
    ["7 years"],
    notes="page 1",
)
single(
    "If I ask Harbor Goods to erase my personal information, how long until it is gone?",
    "data-retention",
    [
        "Once the identity is verified, we complete the deletion within 30 days and send a confirmation email."
    ],
    [2],
    ["30 days"],
    PARA,
    "page 2",
)
single(
    "How long are backups of the order and customer systems kept?",
    "data-retention",
    ["Backups of the order and customer systems are kept for 35 days."],
    [3],
    ["35 days"],
    notes="page 3",
)
single(
    "What is the minimum order value for free shipping?",
    "shipping-policy",
    ["Orders of 75 USD or more ship free by standard service to Zones 1 to 4."],
    [None],
    ["75"],
    EXACT,
)
single(
    "How soon after I buy something does it leave the warehouse?",
    "shipping-policy",
    ["Our handling time is 1 business day for in-stock items."],
    [None],
    ["1 business day"],
    PARA,
    "peak season can extend this, see holiday-schedule",
)
single(
    "How much does a standard parcel to Zone 2 cost?",
    "shipping-rates-zones",
    ["Zone 2 parcels cost 8.40 USD each."],
    [None],
    ["8.40"],
    EXACT,
)
single(
    "What is the remote-area surcharge, and does it apply when shipping is free?",
    "shipping-rates-zones",
    [
        "A remote-area surcharge of 6.50 USD per parcel applies to every parcel sent to a postal code on the remote-area list, even when the order qualifies for free shipping."
    ],
    [None],
    ["6.50"],
    EXACT,
)
single(
    "What is the heaviest single parcel NorthRoute Parcel will carry?",
    "carrier-rules",
    ["NorthRoute Parcel carries single parcels weighing up to 30 lb."],
    [None],
    ["30 lb"],
)
single(
    "Can two promo codes be used on the same order?",
    "promo-price-rules",
    ["Only one promo code can be used per order."],
    [None],
    ["one promo code"],
)
single(
    "Once my returned item has been checked, how soon do I get my money back?",
    "refunds-chargebacks",
    ["Refunds are issued within 5 business days after the returned item passes inspection."],
    [None],
    ["5 business days"],
    PARA,
    "decoy: returns-policy-2025 said 14 business days after arrival",
)
single(
    "Which chargebacks does the finance team accept without contesting them?",
    "refunds-chargebacks",
    [
        "Chargebacks for amounts under 25 USD are accepted without a contest, because the cost of responding is higher than the amount."
    ],
    [None],
    ["25 USD"],
)
single(
    "Who must approve a purchase order worth 12,000 USD?",
    "purchase-orders",
    [
        "Purchase orders above 10,000 USD need approval from both the Operations Manager and the Finance Director."
    ],
    [None],
    ["Finance Director"],
    EXACT,
    "PO threshold 10,000 USD",
)
single(
    "What format do purchase order numbers use?",
    "purchase-orders",
    ["Each PO gets a number in the format PO-2026-0001, counting up through the year."],
    [None],
    ["PO-2026-0001"],
    EXACT,
)
single(
    "What price difference is allowed in the three-way match?",
    "invoice-rules",
    [
        "A price difference is accepted if it is no more than 2 percent of the line value or 50 USD, whichever is lower."
    ],
    [None],
    ["2 percent", "50 USD"],
    EXACT,
)
single(
    "On which day of the week are supplier invoices paid?",
    "invoice-rules",
    ["Payment runs take place every Thursday."],
    [None],
    ["Thursday"],
)
single(
    "Do Harbor Goods gift cards expire?",
    "payment-methods",
    ["Gift card codes start with HGC followed by 12 digits, and gift cards never expire."],
    [None],
    ["never expire"],
)
single(
    "How often are class A SKUs counted?",
    "inventory-counts",
    ["Class A SKUs are counted every week."],
    [None],
    ["every week"],
)
single(
    "When does a stock count difference mean someone else has to count the bin again?",
    "inventory-counts",
    [
        "A variance of more than 3 percent of the system quantity triggers a recount by a second counter."
    ],
    [None],
    ["3 percent"],
    PARA,
)
single(
    "How soon must visible transport damage on a supplier delivery be reported?",
    "receiving-goods",
    [
        "Visible transport damage must be reported to the carrier and the supplier within 48 hours of delivery, with photos of the damaged cartons and contents."
    ],
    [None],
    ["48 hours"],
)
single(
    "How much empty space should a packer leave around the items when picking a box?",
    "packing",
    [
        "Use the smallest box that leaves at least 2 inches of padding between the items and every side of the box."
    ],
    [None],
    ["2 inches"],
    PARA,
)
single(
    "How many photos must a customer send when reporting transit damage?",
    "damaged-goods",
    [
        "The report must include at least 3 photos: one of the outer box, one of the packing inside, and one of the damaged item."
    ],
    [None],
    ["3 photos"],
)
single(
    "What is the first-response target for support emails?",
    "support-hours-sla",
    ["The first-response target for email is 4 business hours."],
    [None],
    ["4 business hours"],
)
single(
    "What is the last order day for standard delivery before Christmas 2026?",
    "holiday-schedule",
    ["The last order day for standard delivery is Wednesday 16 December 2026."],
    [None],
    ["16 December"],
)

# --- multi (10) ------------------------------------------------------------------------------

multi(
    "When is a Trailworks invoice due, and what late fee can Trailworks charge if Harbor Goods pays after that?",
    ["supplier-terms-trailworks", "invoice-rules"],
    [
        "Payment is due net 45 from the invoice date.",
        "Suppliers may charge a late fee of 1.5 percent per month on any invoice amount that Harbor Goods pays after the due date.",
    ],
    [1, None],
    ["net 45", "1.5 percent"],
    EXACT,
    "decoy: Alpinewood is net 30; Trailworks page 1",
)
multi(
    "Who has to approve a purchase order placed at exactly the Alpinewood minimum order value?",
    ["supplier-terms-alpine", "purchase-orders"],
    [
        "The minimum order value is 2,500 USD per purchase order, measured before freight.",
        "Purchase orders above 1,000 USD and up to 10,000 USD need approval from the Operations Manager.",
    ],
    [1, None],
    ["2,500", "Operations Manager"],
    notes="decoy: Trailworks minimum is 4,000 USD; Alpinewood page 1",
)
multi(
    "A customer orders 90 USD of gear by standard service to a Zone 3 postal code on the remote-area list. What do they pay for shipping?",
    ["shipping-policy", "shipping-rates-zones"],
    [
        "Orders of 75 USD or more ship free by standard service to Zones 1 to 4.",
        "A remote-area surcharge of 6.50 USD per parcel applies to every parcel sent to a postal code on the remote-area list, even when the order qualifies for free shipping.",
    ],
    [None, None],
    ["6.50"],
    notes="free shipping applies but the surcharge does not go away",
)
multi(
    "After a change-of-mind return reaches the warehouse, how soon is it inspected and what is deducted from the refund?",
    ["returns-policy", "refunds-chargebacks"],
    [
        "For change-of-mind returns, a flat return label fee of 6.95 USD is deducted from the refund.",
        "Returned items are inspected within 3 business days of arriving at the warehouse.",
    ],
    [None, None],
    ["6.95", "3 business days"],
    notes="decoy: returns-policy-2025 had different return shipping rules",
)
multi(
    "How long is the customer warranty on Harbor Line backpacks, and for how long does Trailworks reimburse those claims?",
    ["warranty", "supplier-terms-trailworks"],
    [
        "Harbor Line backpacks, our own brand, are covered for 5 years from the date of delivery.",
        "Trailworks reimburses Harbor Goods for approved warranty claims on Trailworks products for up to 1 year from delivery to the customer, and for up to 5 years for Harbor Line backpacks.",
    ],
    [1, 3],
    ["5 years"],
    notes="warranty page 1; Trailworks page 3",
)
multi(
    "What is the heaviest carton a packer may send with NorthRoute, and which carrier takes heavier shipments?",
    ["packing", "carrier-rules"],
    [
        "A single carton must not weigh more than 30 lb, so that it can travel with NorthRoute Parcel.",
        "Swiftline Freight carries shipments weighing from 30.1 lb to 150 lb, either as one heavy carton or as several cartons strapped together.",
    ],
    [None, None],
    ["30 lb", "Swiftline"],
)
multi(
    "What is the normal handling time, and how long can it be during the 2026 peak season?",
    ["shipping-policy", "holiday-schedule"],
    [
        "Our handling time is 1 business day for in-stock items.",
        "During peak season, handling time may extend to 2 business days, because order volume is several times higher than usual.",
    ],
    [None, None],
    ["1 business day", "2 business days"],
)
multi(
    "How long does a customer have to report a parcel that arrived broken, and how fast should support first reply to that ticket?",
    ["damaged-goods", "support-hours-sla"],
    [
        "Customers must report transit damage within 7 days of delivery.",
        "Priority tickets get a first response within 2 business hours.",
    ],
    [None, None],
    ["7 days", "2 business hours"],
    PARA,
    "a damaged order is a priority ticket",
)
multi(
    "Which warehouse document must agree with the invoice before accounts payable pays a supplier, and when is it created?",
    ["invoice-rules", "receiving-goods"],
    [
        "The invoice, the purchase order and the receiving report must agree on the items, quantities and prices.",
        "The receiver enters a receiving report in the inventory system on the day of delivery, listing the quantity received in good condition for each SKU.",
    ],
    [None, None],
    ["receiving report"],
)
multi(
    "Are paid supplier invoices and customer order records kept for the same length of time?",
    ["invoice-rules", "data-retention"],
    [
        "Paid invoices and their matching documents are stored in the finance system for 7 years.",
        "Order records, including invoices and payment records, are kept for 7 years after the end of the financial year in which the order was placed, to meet tax and accounting rules.",
    ],
    [None, 1],
    ["7 years"],
    notes="data-retention page 1",
)

# --- unanswerable (10) -----------------------------------------------------------------------

unanswerable(
    "Does Harbor Goods offer a student discount?", ["student"], "no document mentions students"
)
unanswerable("What is the name of the Harbor Goods CEO?", ["CEO"], "no named executives")
unanswerable(
    "How many points do customers earn in the Harbor Goods loyalty program?",
    ["loyalty"],
    "no loyalty program in the documents",
)
unanswerable(
    "Can customers collect their orders by curbside pickup at the warehouse?",
    ["curbside"],
    "no in-person collection described",
)
unanswerable("Does Harbor Goods buy carbon offsets for its shipments?", ["carbon"], "not covered")
unanswerable(
    "What hourly wage does Harbor Goods pay its warehouse packers?", ["wage"], "no pay data"
)
unanswerable(
    "Does Harbor Goods sell bicycles or bicycle parts?", ["bicycle"], "no product range list"
)
unanswerable(
    "Is there a Harbor Goods mobile app for placing orders?", ["mobile app"], "not covered"
)
unanswerable("In what year was Harbor Goods founded?", ["founded"], "no company history")
unanswerable(
    "What reward do customers get for a referral that brings in a new customer?",
    ["referral"],
    "no referral scheme",
)


# ---------------------------------------------------------------------------------------------


def render(d: dict) -> str:
    front = (
        "---\n"
        f"doc_id: {d['doc_id']}\n"
        f"title: {d['title']}\n"
        f"version: {d['version']}\n"
        f"effective: {d['effective']}\n"
        "---\n"
    )
    return front + d["body"] + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> None:
    for d in DOCS:
        suffix = ".source.md" if d["pdf"] else ".md"
        write_text(CORPUS_DIR / f"{d['doc_id']}{suffix}", render(d))

    write_text(QUESTIONS_PATH, json.dumps(QUESTIONS, indent=2) + "\n")

    n_pdf = sum(1 for d in DOCS if d["pdf"])
    types = Counter(x["type"] for x in QUESTIONS)
    tags = Counter(t for x in QUESTIONS for t in x["tags"])
    print(
        f"documents: {len(DOCS)} ({len(DOCS) - n_pdf} markdown, {n_pdf} PDF sources) -> {CORPUS_DIR}"
    )
    print(
        f"questions: {len(QUESTIONS)} (single {types['single']}, multi {types['multi']}, "
        f"unanswerable {types['unanswerable']}) -> {QUESTIONS_PATH}"
    )
    print(f"tags: exact-term {tags['exact-term']}, paraphrase {tags['paraphrase']}")


if __name__ == "__main__":
    main()
