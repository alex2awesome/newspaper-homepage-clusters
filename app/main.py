## this is just for testing purposes to enable quick rendering.

from flask import Flask, render_template, request
from flask.json import jsonify
import datetime
import json
import os, glob

app = Flask(__name__, template_folder='.')

basedir = os.path.abspath(os.path.dirname(__file__))
data_file = os.path.join(basedir, 'data/data-downsampled.json')
with open(data_file) as f:
    input_data = json.load(f)

output_dir = 'data/output_data'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    for f in glob.glob(os.path.join(output_dir, 'output-annotation-*.json')):
        previous_output = json.load(open(f))
        doc_id = previous_output['doc_id']
        if doc_id in input_data:
            input_data[doc_id]['completed'] = True


@app.route('/hello', methods=['GET'])
def main():
    return 'hello world!'

@app.route('/check_data', methods=['GET'])
def return_data():
    return jsonify(input_data)

@app.route('/check_task', methods=['GET'])
def render_task():
    source = request.args.get('source', 'nyt')
    doc_id = request.args.get('doc_id')
    v_x = request.args.get('v_x')
    v_y = request.args.get('v_y')
    html_version = request.args.get('html_version')
    if html_version == 'jsplumb':
        template = 'templates/visualize-doc-level-edits-jsplumb.html'
    else:
        template = 'templates/visualize-doc-level-edits-d3.html'

    instructions_html_file = 'static/assets/instructions.html'
    with open(instructions_html_file) as f:
        instructions_html = f.read()

    if doc_id is not None and v_x is not None and v_y is not None:
        k = str((source, int(doc_id), int(v_x), int(v_y)))
    else:
        import random
        keys = list(input_data.keys())
        if False:
            random.shuffle(keys)
        for k in keys:
            if input_data[k].get('completed', False) == False:
                break

    # get data
    datum = input_data[k]
    nodes = datum['nodes']
    arcs = datum['arcs']
    for n in nodes:
        n['sentence'] = n['sentence'].replace('"', '')
    datum['nodes'] = nodes
    datum['arcs'] = arcs

    return render_template(
        template,
        instructions=instructions_html,
        data=datum,
        doc_id=k,
        do_mturk=False,
        start_time=str(datetime.datetime.now()),
    )


@app.route('/view_task_match', methods=['GET'])
def match_sentences():
    source = request.args.get('source')
    doc_id = request.args.get('doc_id')
    v_x = request.args.get('v_x')
    v_y = request.args.get('v_y')
    if doc_id is not None and v_x is not None and v_y is not None:
        k = str(source, (int(doc_id), int(v_x), int(v_y)))
    else:
        keys = list(input_data.keys())
        for k in keys:
            if input_data[k].get('completed', False) == False:
                break

    # get data
    datum = input_data[k]
    nodes = datum['nodes']
    arcs = datum['arcs']
    for n in nodes:
        n['sentence'] = n['sentence'].replace('"', '')
    datum['nodes'] = nodes
    datum['arcs'] = arcs

    return render_template(
        'templates/match-sentences.html',
        data=datum,
        doc_id=k,
        do_mturk=False,
        start_time=str(datetime.datetime.now()),
    )


@app.route('/view_task_edit', methods=['GET'])
def edit_old_version():
    source = request.args.get('source')
    doc_id = request.args.get('doc_id')
    v_x = request.args.get('v_x')
    v_y = request.args.get('v_y')
    if doc_id is not None and v_x is not None and v_y is not None:
        k = str((source, int(doc_id), int(v_x), int(v_y)))
    else:
        keys = list(input_data.keys())
        import random
        random.shuffle(keys)
        for k in keys:
            if input_data[k].get('completed', False) == False:
                if isinstance(k, str):
                    import ast
                    k = ast.literal_eval(k)
                source, doc_id, v_x, v_y = k
                k = str(k)
                break

    # get data
    datum = input_data[k]
    nodes = datum['nodes']
    v_x = min(list(map(lambda x: x['version'], nodes)))
    nodes = list(filter(lambda x: x['version'] == v_x, nodes))
    for n in nodes:
        n['sentence'] = n['sentence'].replace('"', '')

    return render_template(
        'templates/edit-sentences.html',
        data=nodes,
        doc_id=k,
        do_mturk=False,
        start_time=str(datetime.datetime.now()),
        source=source,
        entry_id=doc_id,
        v_x=v_x,
        v_y=v_y,
    )

@app.route('/post_task', methods=['POST'])
def post_data():
    output_data = request.get_json()
    output_data['end_time'] = str(datetime.datetime.now())

    doc_id = output_data['doc_id']
    input_data[doc_id]['completed'] = True

    ##
    output_file_idx = len(glob.glob(output_dir + '/*'))
    with open(os.path.join(output_dir, 'output-annotation-%s.json' % output_file_idx) , 'w') as f :
        json.dump(output_data, f)
    return "success"


if __name__ == '__main__':
    app.run(debug=True, port=5002)