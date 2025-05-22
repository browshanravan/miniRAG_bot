from google import genai
from google.genai import types


def gemini_llm(project, location, model, credentials, question, contents):
    client = genai.Client(
        vertexai=True,
        project=project,
        location=location,
        credentials= credentials,
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="\n".join(contents))] #since contents preprompt is a list of text
            ),
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question)]
            )
            ]

    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 1,
        seed = 0,
        max_output_tokens = 65535,
        safety_settings = [types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF"
        ),types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF"
        )],
        )

    for chunk in client.models.generate_content_stream(
        model = model,
        contents = contents,
        config = generate_content_config,
        ):
        print(chunk.text, end="")