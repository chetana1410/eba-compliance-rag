from regulations_rag_eval.utils import prompt_escape
from typing import Dict, List, Any


def create_cag_prompt(question: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
  escaped_question = prompt_escape(question)

  prompt_parts = [
    {"type": "text", "text": "#system\nYou are a virtual assistant for the European Banking Authority (EBA), handling user inquiries related to Liquidity Risk regulations. The user's query specifically pertains to Regulation (EU) No. 575/2013 (CRR) or Delegated Regulation (EU) No. 2015/61 (LCR DA)."},
    {"type": "text", "text": "#context:"},
    context,  # Context is placed directly in the array
    {"type": "text", "text": "#task\nAnswer the question based on the instructions below. 1. Analyze the User's Question (#question):\n- Identify the central topic and relevant keywords related to Liquidity Risk and the specified EBA regulations. 2. Leverage the Provided Context (#context):\n- Incorporate the context (including CRR articles and additional information) to tailor the answer to the user's specific scenario.\n3. Liquidity Risk Topic:\n- Reference relevant articles from provided context (#context) that address the specific aspect of Liquidity Risk raised in the question. 4. Desired Answer (#answer):\n- Use only the information provided in the context and knowledge of general legal terms in the European context to answer the question. - Craft a well-reasoned and informative response that covers all aspects of the user's query.\n- Clearly articulate the regulatory implications while considering the provided context.\n- Maintain a professional and informative tone suitable for the EBA."},
    {"type": "text", "text": f"#question: \n{escaped_question}"},
    {"type": "text", "text": "#answer:"}
  ]

  return prompt_parts
