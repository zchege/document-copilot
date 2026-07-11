# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.workflow import Workflow, START
from google.adk.events.event import Event
from google.adk.agents.context import Context
from google.genai import types

# Setup Google Cloud / Vertex AI environment variables if credentials exist
import google.auth
try:
    _, project_id = google.auth.default()
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
except Exception:
    # Fallback to Gemini AI Studio (requires GEMINI_API_KEY or GOOGLE_API_KEY)
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"


# 1. Pydantic model for structured classification output
class QueryClassification(BaseModel):
    category: str = Field(
        description="Must be 'shipping' if the query is related to shipping (rates, tracking, delivery, returns), or 'unrelated' otherwise."
    )


# 2. Save query node to capture original query string
def save_query(node_input: types.Content) -> Event:
    query_text = ""
    if node_input and node_input.parts:
        query_text = "".join(part.text for part in node_input.parts if part.text)
    return Event(output=query_text, state={"original_query": query_text})


# 3. Classifier agent using structured output
classifier_agent = LlmAgent(
    name="classifier_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are an assistant that classifies incoming user queries. "
        "Determine if the query is related to shipping (rates, tracking, delivery, returns) or unrelated. "
        "Provide your classification matching the required schema."
    ),
    output_schema=QueryClassification,
)


# 4. Routing node to dispatch path and forward original query
def router(ctx: Context, node_input: dict) -> Event:
    category = node_input.get("category", "unrelated")
    original_query = ctx.state.get("original_query", "")
    if category == "shipping":
        return Event(output=original_query, route="shipping")
    else:
        return Event(output=original_query, route="unrelated")


# 5. FAQ specialist agent node
faq_agent = LlmAgent(
    name="faq_agent",
    model=Gemini(
        model="gemini-2.5-flash",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a helpful customer support representative for a shipping company. "
        "Answer the customer's shipping-related questions (rates, tracking, delivery, returns) "
        "politely, clearly, and concisely."
    ),
)


# 6. Polite decline function node
def polite_decline(node_input: str) -> Event:
    decline_message = (
        "I'm sorry, but I can only assist with shipping-related inquiries "
        "(such as rates, tracking, delivery, and returns). "
        "How can I help you with your shipping needs today?"
    )
    return Event(
        output=decline_message,
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=decline_message)]
        )
    )


# 7. Workflow definition
root_agent = Workflow(
    name="customer_support_workflow",
    edges=[
        (START, save_query),
        (save_query, classifier_agent),
        (classifier_agent, router),
        (router, faq_agent, "shipping"),
        (router, polite_decline, "unrelated"),
    ],
)

# 8. Application instance
app = App(
    root_agent=root_agent,
    name="app",
)
