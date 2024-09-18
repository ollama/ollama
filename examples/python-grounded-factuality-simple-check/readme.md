# Simple Bespoke-Minicheck Example

`bespoke-minicheck` is a model for checking if a claim is supported by a document. It is used through the **generate** endpoint, which is called in this example with a `prompt` that includes the expected formatting of the user input. 

## Running the Example

1. Ensure you have the `bespoke-minicheck` model installed:

   ```bash
   ollama pull bespoke-minicheck
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the program:

   ```bash
   python main.py
   ```

4. Enter a document and a claim when prompted:

   ```bash
   Enter a document: Roses are red.

   Enter a claim: Roses are blue. 
   ```

   The claim and document are then given to the `bespoke-minicheck` as inputs, which then generates a response (Yes or No) on whether the claim is supported by the document.

   ```bash
   Is the claim supported by the document according to bespoke-minicheck? No
   ```

## More Examples

Document ([source](https://en.wikipedia.org/wiki/Apple_I)): 
> The Apple Computer 1 (Apple-1[a]), later known predominantly as the Apple I(written with a Roman numeral),[b] is an 8-bit motherboard-only personal computer designed by Steve Wozniak[5][6] and released by the Apple Computer Company (now Apple Inc.) in 1976. The company was initially formed to sell the Apple I – its first product – and would later become the world's largest technology company.[7] The idea of starting a company and selling the computer came from Wozniak's friend and Apple co-founder Steve Jobs.[8][9] One of the main innovations of the Apple I was that it included video display terminal circuitry on its circuit board, allowing it to connect to a low-cost composite video monitor or television, instead of an expensive computer terminal, compared to most existing computers at the time.

Claim: 
>The Apple I is a 16-bit computer.

Expected output:
>Is the claim supported by the document according to bespoke-minicheck? **No**

Claim: 
>Apple was originally called the Apple Computer Company.

Expected output:
>Is the claim supported by the document according to bespoke-minicheck? **Yes**
