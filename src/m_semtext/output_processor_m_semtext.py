import os

from bs4 import BeautifulSoup
import pandas as pd
from typing import List

from m_semtext.dataset_m_semtext import MSemTextDataset


class MSemTextOutputProcessor:

    def __init__(self, prediction_col_name: str = "html/prediction"):
        self.prediction_col_name = prediction_col_name

    def process(self, predictions: List[List[List[int]]], dataset: MSemTextDataset, prediction_col_name: str = None) -> pd.DataFrame:
        if prediction_col_name is None:
            prediction_col_name = self.prediction_col_name
        dataset_df = dataset.dataset_df
        predictions = [pred for outer_list in predictions for inner_list in outer_list for pred in inner_list]
        dataset_df[prediction_col_name] = predictions
        return dataset_df

    def save_to_csv(self, file_path: str, predictions: List[List[List[int]]], dataset: MSemTextDataset,
                    prediction_col_name: str = None,) -> None:
        df = self.process(predictions, dataset, prediction_col_name)
        df.to_csv(file_path)


class MSemTextHTMLLabeler:

    DEFAULT_SEMANTIC_NAME_TAG_MAP = {
        "primary headline": "h1",
        "secondary headline": "h2",
        "tertiary headline": "h3",
        "quaternary headline": "h4",
        "quinary headline": "h5",
        "senary headline": "h6",
        "navigation": "nav",
        "section": "section",
        "menu item": "menuitem",
        "block quotation": "blockquote",
        "description details": "dd",
        "description list": "dl",
        "description term": "dt",
        "division": "div",
        "figure caption": "figcaption",
        "paragraph": "p",
        "list item": "li",
        "ordered list": "ol",
        "unordered list": "ul",
        "anchor": "a"
    }

    def __init__(self, label_attribute_name: str = "predicted_label", semantic_name_tag_map: dict = None,
                 label_col_name: str = "html/prediction", html_file_name_col_name: str = "html/originalId",
                 class_sequence_col_name: str = "html/classSequence", tag_sequence_col_name: str = "html/tagSequence",
                 text_sequence_col_name: str = "html/textSequence", html_str_col_name: str = "html/html"):
        if semantic_name_tag_map is None:
            self.semantic_name_tag_map = self.DEFAULT_SEMANTIC_NAME_TAG_MAP
        else:
            self.semantic_name_tag_map = semantic_name_tag_map
        self.label_attribute_name = label_attribute_name
        self.label_col_name = label_col_name
        self.html_file_name_col_name = html_file_name_col_name
        self.class_sequence_col_name = class_sequence_col_name
        self.tag_sequence_col_name = tag_sequence_col_name
        self.text_sequence_col_name = text_sequence_col_name
        self.html_str_col_name = html_str_col_name

    def label_htmls(self, htmls_dir_path: str = None, htmls_df: pd.DataFrame = None, dataset_df: pd.DataFrame = None, dataset_file_path: str = None,
                    labeled_htmls_dir_path: str = None):
        if htmls_dir_path and htmls_df:
            raise ValueError("Only one of 'htmls_dir_path' and 'htmls_df' can be used!")
        if htmls_dir_path is None and htmls_df is None:
            raise ValueError("One of 'htmls_dir_path-path' and 'htmls_df' must not be empty!")
        if dataset_df is None:
            dataset_df = pd.read_csv(dataset_file_path)
        if labeled_htmls_dir_path is None:
            labeled_htmls_dir_path = htmls_dir_path
        if not os.path.exists(labeled_htmls_dir_path):
            os.makedirs(labeled_htmls_dir_path)

        dataset_df[self.tag_sequence_col_name] = dataset_df[self.tag_sequence_col_name].str[1:-1].str.replace(", ", ",").str.split(",")
        dataset_df[self.text_sequence_col_name] = dataset_df[self.text_sequence_col_name].str[1:-1].str.replace(", ", ",").str.split(",")
        dataset_df[self.class_sequence_col_name] = dataset_df[self.class_sequence_col_name].str[1:-1].str.replace(", ", ",").str.split(",")

        labeled_htmls = []
        for html_file_name in dataset_df[self.html_file_name_col_name].unique():
            print(html_file_name)
            html_file_path = os.path.join(htmls_dir_path, html_file_name)
            labeled_html_file_path = os.path.join(labeled_htmls_dir_path, html_file_name)
            html_df = dataset_df[dataset_df[self.html_file_name_col_name] == html_file_name]
            if htmls_df:
                html_str = htmls_df[htmls_df[self.html_file_name_col_name] == html_file_name][self.html_str_col_name]
                labeled_html = self.label_html(html_df, html_str=html_str)
            else:
                labeled_html = self.label_html(html_df, html_file_path=html_file_path,
                                               labeled_html_file_path=labeled_html_file_path)
            labeled_htmls.append((html_file_name, labeled_html))
        return pd.DataFrame(labeled_htmls, columns=[self.html_file_name_col_name, self.html_str_col_name])

    def label_html(self, html_df: pd.DataFrame, html_file_path: str = None, html_str: str = None, labeled_html_file_path: str = None):
        if html_file_path and html_str:
            raise ValueError("Only one of 'html_file_path' and 'html_str' can be used!")
        if html_file_path is None and html_str is None:
            raise ValueError("One of 'html_file-path' and 'html_str' must not be empty!")
        if html_file_path:
            with open(html_file_path) as f:
                html = BeautifulSoup(f, "html.parser")
        if html_str:
            html = BeautifulSoup(html_str, "html.parser")
        if labeled_html_file_path is None:
            labeled_html_file_path = html_file_path

        for index, row in html_df.iterrows():
            label = row[self.label_col_name]
            semantic_tag = row[self.tag_sequence_col_name][-1]
            tag = self.semantic_name_tag_map.get(semantic_tag, semantic_tag)
            text = row[self.text_sequence_col_name][-1]
            text = self._clean_text(text)

            # first, check if there is an element with the text block's actual tag
            # and that element contains the text in its own element.
            elements = html.select(f'{tag}:-soup-contains-own("{text}")')

            # if there is no element with the above criteria, try to look for element with the parent's tag
            # and contains the text in itself or any of its descendants.
            if len(elements) == 0:
                semantic_tag = row[self.tag_sequence_col_name][-2]
                tag = self.semantic_name_tag_map.get(semantic_tag, semantic_tag)
                elements = html.select(f'{tag}:-soup-contains("{text}")')

            if len(elements) != 1:
                # if the number of elements is not 1, try to search for element that has the parent tag followed by the tag,
                # but contains the text in itself or any of its descendants.
                semantic_tag = row[self.tag_sequence_col_name][-1]
                tag = self.semantic_name_tag_map.get(semantic_tag, semantic_tag)
                parent_semantic_tag = row[self.tag_sequence_col_name][-2]
                parent_tag = self.semantic_name_tag_map.get(parent_semantic_tag, parent_semantic_tag)
                elements = html.select(f'{parent_tag} > {tag}:-soup-contains("{text}")')

            if len(elements) != 1:
                # if the number of elements is still not 1, try to search exactly with all of the tags in the text block,
                # that contains the text
                tags = [self.semantic_name_tag_map.get(sem_tag, sem_tag) for sem_tag in row[self.tag_sequence_col_name]]
                tags_str = ' '.join(tags)
                elements = html.select(f'{tags_str}:-soup-contains("{text}")')

            if len(elements) != 1:
                # if the number of elements is still not 1, try to search for element that has the tag and
                # contains the whole text from the text block combined
                semantic_tag = row[self.tag_sequence_col_name][-1]
                tag = self.semantic_name_tag_map.get(semantic_tag, semantic_tag)
                text = ', '.join(row[self.text_sequence_col_name])
                text = self._clean_text(text)
                elements = self.__find_elements_with_combined_text(html, tag, text)

            if len(elements) != 1:
                # if the number of elements is still not 1, try to search for element that has the parent tag and
                # contains the whole text from the text block combined
                parent_semantic_tag = row[self.tag_sequence_col_name][-2]
                parent_tag = self.semantic_name_tag_map.get(parent_semantic_tag, parent_semantic_tag)
                text = ', '.join(row[self.text_sequence_col_name])
                text = self._clean_text(text)
                elements = self.__find_elements_with_combined_text(html, parent_tag, text)

            # if len(elements) != 1:
            #     print(index)
            #     print(elements)
            #     print(tag)
            #     print(text)
            for element in elements:
                element[self.label_attribute_name] = label
            # if len(elements) == 1:
            #     elements[0][self.label_attribute_name] = label
            # print("=======")

        with open(labeled_html_file_path, "w") as f:
            f.write(str(html))

        return html

    def _clean_text(self, text: str):
        if text.startswith('"'):
            text = text[1:]
        if text.endswith('"'):
            text = text[:-2]
        text = text.replace("(", "\\(").replace(")", "\\)").replace("'", "\\'").replace('"', '\\"')
        return text

    def __find_elements_with_combined_text(self, html: BeautifulSoup, tag: str, text: str):
        elements = []
        tag_elements = html.select(tag)
        for tag_element in tag_elements:
            tag_element_text = self._clean_text(' '.join(' '.join(tag_element.stripped_strings).strip().split()))
            # check equality regardless of whitespace
            if ''.join(tag_element_text.split()) == ''.join(text.split()):
                elements.append(tag_element)
        return elements


# if __name__ == "__main__":
#     labeler = MSemTextHTMLLabeler(label_col_name="html/label")
#     df = labeler.label_htmls("../../dataset/init-processed-dataset/dataset-googletrends-2017-dev",
#                         dataset_file_path="../../dataset/html-labeler-test.csv", labeled_htmls_dir_path="../../dataset/labeled-test")
#     print(df)