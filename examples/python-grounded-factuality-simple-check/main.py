"""Simple example to demonstrate how to use the bespoke-minicheck model."""

import ollama

# NOTE: ollama must be running for this to work, start the ollama app or run `ollama serve`


def check(document, claim):
    """Checks if the claim is supported by the document by calling bespoke-minicheck.

    Returns Yes/yes if the claim is supported by the document, No/no otherwise.
    Support for logits will be added in the future.

    bespoke-minicheck's system prompt is defined as:
      'Determine whether the provided claim is consistent with the corresponding
      document. Consistency in this context implies that all information presented in the claim
      is substantiated by the document. If not, it should be considered inconsistent. Please
      assess the claim's consistency with the document by responding with either "Yes" or "No".'

    bespoke-minicheck's user prompt is defined as:
      "Document: {document}\nClaim: {claim}"
    """
    prompt = f"Document: {document}\nClaim: {claim}"
    response = ollama.generate(
        model="bespoke-minicheck", prompt=prompt, options={"num_predict": 2, "temperature": 0.0}
    )
    return response["response"].strip()


def get_user_input(prompt):
    user_input = input(prompt)
    if not user_input:
        exit()
    print()
    return user_input


def main():
    while True:
        # Get a document from the user (e.g. "Ryan likes running and biking.")
        document = get_user_input("Enter a document: ")
        # Get a claim from the user (e.g. "Ryan likes to run.")
        claim = get_user_input("Enter a claim: ")
        # Check if the claim is supported by the document
        grounded_factuality_check = check(document, claim)
        print(
            f"Is the claim supported by the document according to bespoke-minicheck? {grounded_factuality_check}"
        )
        print("\n\n")


if __name__ == "__main__":
    main()
