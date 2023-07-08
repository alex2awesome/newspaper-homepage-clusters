import uuid, re
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import util_newssniffer_parsing as unp
import os
import subprocess
import html

class Switch():
    def __init__(self, categories=[]):
        self.switch = {}
        self.categories = categories
        for i in self.categories:
            self.switch[i] = False

    def toggle(self, i):
        for j in self.categories:
            self.switch[j] = False
        self.switch[i] = True

    def get_on(self):
        for k, v in self.switch.items():
            if v == True:
                return k
        return False

    def get_level_in(self):
        key = self.get_on()
        if key:
            return int(re.search('\d', key)[0])


class InstructionsHandler():
    def __init__(self, path_to_img_src='.'):
        self.path_to_img_src = path_to_img_src
        self.img_count = 0
        self.switch = Switch(categories=['in_h2', 'in_h3', 'in_h4', 'in_h5'])
        self.elem_dict = {
            'new_soup': BeautifulSoup(features='lxml'),
            'h2_div': None,
            'h3_div': None,
            'h4_div': None,
            'h5_div': None,
            'card_body_div': None,
            'card_body_content_div': None
        }
        self.card_id_count = 0

    def _create_div_of_same_type(self, elem, class_name=None, **kwargs):
        div = self.elem_dict['new_soup'].new_tag('div')
        if class_name is not None:
            div.attrs['class'] = class_name
        div.append(elem)
        return div

    def _format_image(self, img_elem, checkbox_id=None):
        # checkbox html:
        # <div class="form-check form-switch">
        #  <input class="form-check-input" type="checkbox" id="flexSwitchCheckDefault">
        #  <label class="form-check-label" for="flexSwitchCheckDefault">Default switch checkbox input</label>
        # </div>

        # create image div
        img_div = self.elem_dict['new_soup'].new_tag('div')
        img_div.attrs['class'] = 'example-image'

        if img_elem.name != 'img':
            img_elem = img_elem.find('img')
        img_elem.attrs['src'] = os.path.join(self.path_to_img_src, img_elem.attrs['src'])
        img_elem.attrs['width'] = '100%'
        img_elem.attrs['class'] = 'hidden'

        # create checkbox
        checkbox_div = self.elem_dict['new_soup'].new_tag('div')
        checkbox_div.attrs['class'] = 'form-check form-switch'
        checkbox_input = self.elem_dict['new_soup'].new_tag('input')
        checkbox_input.attrs['class'] = 'form-check-input image-show'
        checkbox_input.attrs['type'] = 'checkbox'
        if checkbox_id is None:
            checkbox_id = 'form-check-input-%s' % self.card_id_count
        checkbox_input.attrs['id'] = checkbox_id

        checkbox_label = self.elem_dict['new_soup'].new_tag('label')
        checkbox_label.attrs['class'] = 'form-check-label'
        checkbox_label.attrs['for'] = checkbox_id
        checkbox_label.insert(0, NavigableString("Show example image"))

        checkbox_div.append(checkbox_input)
        checkbox_div.append(checkbox_label)

        img_div.append(checkbox_div)
        img_div.append(img_elem)


        return img_div

    def _format_example_block(self, elem):
        ex_block = self.elem_dict['new_soup'].new_tag('div')
        ex_block.attrs['class'] = 'example-block'
        ex_text = elem.get_text().replace('```', '').split('|||')
        ex_text = list(map(lambda x: x.strip(), ex_text))
        if len(ex_text) == 2:
            if ((ex_text[0] != '') and (ex_text[1] != '')):
                html_old, html_new = unp.html_compare_sentences(ex_text[0], ex_text[1])
                text_span = '<div class="example_left_block"><span>' + html_old + '</span></div>'
                text_span += '<span>&rarr;</span>'
                text_span += '<div class="example_right_block"><span>' + html_new + "</span></div>"

                text_span = '<span>' + html_old + '&rarr;' + html_new + '</span>'
            elif ex_text[1] == '':
                text_span = '<span style="background-color:rgba(255,0,0,0.3)">' + ex_text[0] + "</span>"
            else:
                text_span = '<span style="background-color:rgba(0,255,0,0.3)">' + ex_text[1] + '</span>'
        else:
            text_span = '<span>' + ex_text[0] + '</span>'
        span = BeautifulSoup(text_span, 'lxml').span
        ex_block.append(span)
        return ex_block

    def _create_second_level_header_card(self, elem, link_id=None):
        if link_id is None:
            link_id = 'second-level-card-%s' % self.card_id_count
            self.card_id_count += 1

        self.elem_dict['h3_div'] = self.elem_dict['new_soup'].new_tag('div')
        self.elem_dict['h3_div'].attrs['class'] = "second-level-category card"

        card_header_div = self.elem_dict['new_soup'].new_tag('div')
        card_header_div.attrs['class'] = "card-header"

        button_header = self.elem_dict['new_soup'].new_tag('button')
        button_header.attrs['class'] = 'btn btn-link collapsed'
        button_header.attrs['data-bs-toggle'] = 'collapse'
        button_header.attrs['type'] = 'button'
        button_header.attrs['href'] = '#' + link_id
        button_header.attrs['aria-expanded'] = "false"

        button_header.append(elem)
        card_header_div.append(button_header)
        self.elem_dict['h3_div'].append(card_header_div)

        self.elem_dict['card_body_div'] = self.elem_dict['new_soup'].new_tag('div')
        self.elem_dict['card_body_div'].attrs['id'] = link_id
        self.elem_dict['card_body_div'].attrs['class'] = 'collapse'
        self.elem_dict['card_body_div'].attrs['data-parent'] = '#accordion'

        self.elem_dict['card_body_content_div'] = self.elem_dict['new_soup'].new_tag('div')
        self.elem_dict['card_body_content_div'].attrs['class'] = 'card-body'
        return

    def _append_not_null(self, parent, child):
        parent_elem = self.elem_dict[parent]
        child_elem = self.elem_dict[child]

        child_text = child_elem.get_text().strip()
        if (not child_text in ['', 'Example:', 'Explanation:']) or (child_elem.find('img') is not None):
            parent_elem.append(child_elem)

    def _handle_rollup(self, level_in, level_to_stop=2):
        for i in range(level_in, level_to_stop, -1):
            if i == 5:
                self._append_not_null('h4_div', 'h5_div')
            if i == 4:
                self._append_not_null('card_body_content_div', 'h4_div')
            if i == 3:
                self._append_not_null('card_body_div', 'card_body_content_div')
                self._append_not_null('h3_div', 'card_body_div')
                self._append_not_null('h2_div', 'h3_div')
            if i == 2:
                self._append_not_null('new_soup', 'h2_div')

    def generate(self, elems):
        #   The card output div has to be of the format:
        #     <div class="top-level-category">
        #         <h2>Clarification</h2>
        #         <div class="second-level-category card">
        #             <div class="card-header">
        #                 <button class="btn btn-link collapsed" data-bs-toggle="collapse" type="button" href="#test" aria-expanded="false" aria-controls="collapseFour">
        #                     <h3> Title </h3>
        #                 </button> </div>
        #             <div id="test" class="collapse" data-parent="#accordion">
        #                 <div class="card-body">
        #                     <div class="definition"><h4>Definition</h4><p>...</p></div>
        #                     <div class="example"><h4>Example</h4><div class="example-block"><p>...</p></div></div>
        #                     <div class="explanation"><h5>Explanation</h5><p>...</p></div>
        #                 </div>
        #             </div>
        #         </div>
        #    </div>

        for elem in elems:
            if elem.name == 'h2':
                level_in = self.switch.get_level_in()
                if level_in is not None:
                    self._handle_rollup(level_in, level_to_stop=1)
                self.elem_dict['h2_div'] = self._create_div_of_same_type(elem, 'top-level-category')
                self.switch.toggle('in_h2')

            elif elem.name == 'h3':
                level_in = self.switch.get_level_in()
                # end the card, wrap it all up
                if level_in >= 3:
                    self._handle_rollup(level_in, level_to_stop=2)
                self._create_second_level_header_card(elem)
                self.switch.toggle('in_h3')

            elif elem.name == 'h4':
                level_in = self.switch.get_level_in()
                if level_in >= 4:
                    self._handle_rollup(level_in, level_to_stop=3)
                class_name = elem.text.replace(':', '').lower()
                self.elem_dict['h4_div'] = self._create_div_of_same_type(elem, class_name)
                self.switch.toggle('in_h4')

            # child-most elements
            elif elem.name == 'h5':
                self.elem_dict['h5_div'] = self._create_div_of_same_type(elem, 'explanation')
                self.switch.toggle('in_h5')

            elif '```' in elem.get_text():
                example_block_div = self._format_example_block(elem)
                self.elem_dict['h4_div'].append(example_block_div)

            elif elem.find('img') is not None:
                img_div = self._format_image(elem)
                self.elem_dict['h4_div'].append(img_div)

            else:
                level_in = self.switch.get_level_in()
                self.elem_dict[f'h{level_in}_div'].append(elem)

        self._handle_rollup(level_in, level_to_stop=2)
        self.elem_dict['new_soup'].append(self.elem_dict['h2_div'])
        return self.elem_dict['new_soup']


def get_elements_from_soup(soup):
    for a in soup.find_all('a'):
        a.extract()
    elems = []
    for x in soup.body.children:
        if not isinstance(x, Tag):
            continue
        if ('localhost' in x.get_text()):
            continue
        elems.append(x)
    return elems


def download_file(real_file_id):
    """Downloads a file
    Args:
        real_file_id: ID of the file to download
    Returns : IO object with location.

    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
    """
    import io
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload

    creds, _ = google.auth.default()

    try:
        # create drive api client
        service = build('drive', 'v3', credentials=creds)

        file_id = real_file_id

        # pylint: disable=maybe-no-member
        request = service.files().export_media(
            fileId=file_id,
            mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print(F'Download {int(status.progress() * 100)}.')

    except HttpError as error:
        print(F'An error occurred: {error}')
        return

    return file.getvalue()


if __name__ == '__main__':
    import argparse

    img_version_num = 'v1.0.1'
    DEFAULT_GDRIVE_FILE_ID = "1Lv_8WnvdsrbWao4VRs96fL7cTEcBBX9GEaU4f7LHdxc"
    DEFAULT_ORIG = "../app/static/assets/NewsEdits++ MTurk Instructions.docx"
    DEFAULT_IMG_DIR = f"../app/static/assets/img-{img_version_num}"
    DEFAULT_IN = "../app/static/assets/NewsEdits++ MTurk Instructions.html"
    DEFAULT_OUT = "../app/static/assets/instructions.html"
    DEFAULT_OUTPUT_DIR = f"../app/static/assets/img-{img_version_num}/"
    DEFAULT_REL_PATH_TO_IMG = f"/static/assets/img-{img_version_num}"
    # MTURK_PATH_TO_IMG = f"https://cdn.jsdelivr.net/gh/alex2awesome/edit-intentions@master/app/static/assets/img-{img_version_num}/"
    MTURK_PATH_TO_IMG = f"https://cdn.jsdelivr.net/gh/alex2awesome/edit-intentions-pub@master/static/assets/img/"

    parser = argparse.ArgumentParser()
    parser.add_argument('-g_id', help="Google drive ID", type=str, default=DEFAULT_GDRIVE_FILE_ID)
    parser.add_argument('--do_download', help="whether to download or use cached file", action='store_true')
    parser.add_argument('-i', help='Input docx file', type=str, default=DEFAULT_ORIG)
    parser.add_argument('-d', help='Img dir', type=str, default=DEFAULT_IMG_DIR)
    parser.add_argument('-o', help='Output File', type=str, default=DEFAULT_OUT)
    parser.add_argument('--img_path', type=str, default=MTURK_PATH_TO_IMG)

    ##
    args = parser.parse_args()
    data_dir = os.path.dirname(args.d)
    docx_filename = args.i
    if args.do_download:
        file = download_file(args.g_id)
        assert file is not None
        with open(docx_filename, 'wb' ) as f:
            f.write(file)

    html_filename = os.path.basename(docx_filename).replace('.docx', '.html')
    filepath = os.path.join(data_dir, html_filename)
    if not os.path.exists(args.d):
        os.makedirs(args.d)
    subprocess.run(['mammoth', docx_filename, '--output-dir=%s' % args.d])
    subprocess.run(['mv', os.path.join(args.d, html_filename), data_dir])

    with open(filepath, "r") as f:
        html_str = f.read()
        html_str = html.unescape(html_str)

        soup = BeautifulSoup(html_str, 'lxml')
        elems = get_elements_from_soup(soup)
        ih = InstructionsHandler(path_to_img_src=args.img_path)
        generated_html = ih.generate(elems)

        with open(args.o, 'w') as f:
            f.write(str(generated_html))