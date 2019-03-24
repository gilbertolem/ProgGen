# ProgGen

Jazz progression generator using Recurrent Neural Networks.

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

## Principal authors and styles

As a reference, this is a list of the principal authors and styles that can be chosen for the filters:

### Authors

This list was taken from the authors that had >=10 tunes in the iReal app.

- Antonio-Carlos Jobim
- Bill Evans
- Billy Strayhorn
- Charlie Parker
- Chick Corea
- Cole Porter
- Dizzy Gillespie
- Duke Ellington
- George Gershwin
- Harold Arlen
- Harry Warren
- Herbie Hancock
- Hoagy Carmichael
- Horace Silver
- Irving Berlin
- Jerome Kern
- Jimmy McHugh
- Jimmy Van-Heusen
- Joe Henderson
- John Coltrane
- Jule Styne
- Keith Jarrett
- Miles Davis
- Pat Metheny
- Richard Rodgers
- Thelonious Monk
- Wayne Shorter
- Sonny Rollins

### Styles

More than music style, this refers to the default backing track style set in the iReal app. This is the list of the first 10 most common styles:

- Ballad
- Bossa Nova
- Even 8ths
- Latin
- Funk
- Medium Swing
- Medium Up Swing
- Slow Swing
- Up Tempo Swing
- Waltz