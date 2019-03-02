# ProgGen

Jazz progression generator using Recurrent Neural Networks in pytorch.

This code takes ~1500 progression charts in MusicXML format (most of them jazz tunes) and generates new progressions taking them as a reference. The tunes and the progressions can be found in the *iReal Pro* app.

It uses RNNs for learning the structure of the tune dataset.

It can give more importance to some author or style. For example, one could train the network to generate tunes that are 50% Charlie Parker, 40% Thelonious Monk and 10% the rest of the composers, or 60% Ballad and 40% Bossa Nova. The names of the authors and styles can be found in the *iReal Pro* app.

## Usage

One can open the [main.ipnyb](main.ipynb) file to run the code.

If using Google Colab, one can use the **Set environment** section for installing pytorch and cloning the git repository.

In the **Create training data** section, one can specify which authors or tunes wants the net to be trained on. The `filter_names` list is used for the names of the author or styles, and the `filter_fracs` for the desired fractions. For the example of 50% Charlie Parker, 40% Thelonious Monk and 10% the rest of the composers, one could put:
```
filter_names = ['Charlie Parker', 'Thelonious Monk']
filter_fracs = [0.5, 0.4]
```

As the sum of the specified fractions sums to 0.9, the rest is going to be taken from the remaining authors.
