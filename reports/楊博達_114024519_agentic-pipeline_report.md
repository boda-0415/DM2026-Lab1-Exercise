# Your report

## 1. Dataset Overview and Data Quality

This analysis applied an agentic data mining pipeline to four categories of the 20 Newsgroups dataset: `alt.atheism`, `comp.graphics`, `sci.med`, and `soc.religion.christian`. The dataset contained 2,257 documents, with 480, 584, 594, and 599 documents in the respective categories. Initial inspection (`load_dataset_1`) found no missing values or duplicate records. Document lengths varied substantially, with a mean of approximately 1,992.96 characters and a median of 1,235 characters, indicating a right-skewed distribution with several unusually long documents.

## 2. Document-Term Matrix and Feature Filtering

A document-term matrix (DTM) was constructed (`build_dtm_2`) with 2,257 documents and 35,788 terms. The matrix was highly sparse, with approximately 99.55% of its entries equal to zero. Variance filtering (`variance_filter_3`) using a threshold of 0.01 retained 5,354 terms and removed 30,434 low-variance terms. Pearson correlation filtering (`pearson_filter_4`) examined associations between terms and the `comp.graphics` category using a threshold of 0.1, retaining 189 terms. Positively associated terms included `thanks`, `keywords`, `vga`, `video`, and `card`. These results illustrate how feature filtering can reduce dimensionality and highlight terms associated with a particular document category.

## 3. Feature Relationships and Frequent Pattern Mining

Feature correlation analysis (`feature_corr_5`) produced a 20-by-20 correlation matrix and a corresponding heatmap. Some common terms were strongly correlated, including `the` and `of`, suggesting that frequently occurring words may share similar document-level frequency patterns. Frequent pattern mining (`mine_patterns_6`) was performed on the `comp.graphics` category using FP-Growth, variance-based filtering, and a minimum support count of 30. The algorithm identified 61 frequent itemsets. For example, `organization` appeared in 560 documents, while the pair `newsreader` and `tin` appeared in 52 documents. The dominance of terms such as `organization` suggests that email or newsgroup header information may influence the discovered patterns, rather than all frequent patterns representing meaningful topical relationships.

![Feature correlation heatmap](../plots/楊博達_agentic-pipeline_plot_feature_corr_5.png)

## 4. Dimensionality Reduction and Label Binarization

Principal Component Analysis (PCA) was performed with two components (`dim_red_7`). The resulting coordinate matrix contained 997 observations and two dimensions. The first principal component explained approximately 65.94% of the variance, while the second explained 4.33%, giving a combined explained variance ratio of 70.27%. Since the original dataset contained 2,257 documents, the difference in observation counts should be investigated before interpreting the PCA visualization as representing the entire dataset. Label binarization (`binarize_labels_8`) successfully converted the four document categories into a one-hot encoded matrix of shape 2,257 by 4.

![PCA visualization](../plots/楊博達_agentic-pipeline_plot_dim_red_7.png)

## 5. Document Similarity and Conclusion

Finally, cosine similarity was calculated between documents 0 and 1 using raw DTM term counts (`cosine_3`). The resulting similarity score was approximately 0.3532. Both documents concerned computer graphics, but one discussed image-format conversion for printing, while the other focused on geometric meshes and Bézier trimming curves. The score suggests partial overlap in their word-frequency representations without indicating that the documents discuss identical subjects. Overall, the pipeline demonstrated a complete workflow for text exploration, feature selection, pattern discovery, dimensionality reduction, and document similarity analysis. Important limitations include the influence of newsgroup header terms and the need to verify the number of observations used in PCA.*
