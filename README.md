# Cognitive Mapping

The data consists of text and relations, which have three parts (Concept1, Explanation and Concept2). 
Each of the three parts correspond to a multi-word phrase in a text. 

All relations and text can be easily read into a Jupyter notebook.

The goal is to identify the three parts of the relation in a text automatically.

## Example: 
'3-2: <span style="background-color: lightblue;">[concept Giving to the ECB the ultimate responsibility for 
supervision of banks in the euro area concept]</span> <span style="background-color: pink;">[explanation will decisively
 contribute to increase explanation]</span> <span style="background-color: lightblue;">[concept confidence between the 
 banks concept]</span> <span style="background-color: pink;">[explanation and in this way increase explanation]</span> 
 <span style="background-color: lightblue;">[concept the financial stability in the euro area concept]</span>. 
 The euro area governments and the European institutions, including naturally the European Commission and the ECB, 
 will do whatever is necessary to secure the financial stability of the euro area.\n'

## Plan
1. machine learning of paragraphs: do they contain a causal relation or not
2. find phrases of relations in text: either concepts, explanations or not present in a relation
3. identify relations based on recognized concept phrases and explanation phrases
 
We need a tagger or a entity recognition program, for example transformers, huggingface/bert: https://github.com/huggingface/transformers , or Spacy

## Contributing
Contributons are welcome, see [CONTRIBUTING](CONTRIBUTING.md).

## Citation
If you want to cite this software, please use the metdata in [CITATION.cff](CITATION.cff).

## References

Hosseini, M.J., Chambers, N., Reddy, S., Holt, X R., Cohen, S B., Johnson, M., & Steedman, M. (2018). Learning Typed Entailment Graphs with Global Soft Constraints, Transactions of the Association for Computational Linguistics. Sizov, G. & Ozturk, P. (2013). 

Zornitsa Kozareva, Irina Matveeva, Gabor Melli, Vivi Nastase Zornitsa Kozareva, Irina Matveeva, Gabor Melli, Vivi Nastase (2013). Automatic Extraction of Reasoning Chains from Textual Reports. Proceedings of TextGraphs-8 Workshop “Graph-based Methods for Natural Language Processing”, Empirical Methods in Natural Language Processing. 

Noah Jadallah (2021). [Cause-Effect Detection for Software Requirements Based on Token Classification with BERTCause-Effect Detection for Software Requirements Based on Token Classification with BERT](
https://colab.research.google.com/drive/14V9Ooy3aNPsRfTK88krwsereia8cfSPc?usp=sharing#scrollTo=H_kiqxjbW3lh). Seminar Natural Language Processing for Software Engineering, Winter-term 2020/2021, Technical University Munich.

Erik Tjong Kim Sang and Katja Hofmann (2009). [Lexical Patterns or Dependency Patterns: Which Is Better for Hypernym Extraction?](https://ifarm.nl/erikt/papers/conll2009.pdfhttps://ifarm.nl/erikt/papers/conll2009.pdf) In: Proceedings of CoNLL-2009, Boulder, CO, USA, 2009, pages 174-182.
