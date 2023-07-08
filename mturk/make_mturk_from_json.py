import argparse
import os
import random
import json
import shutil
import copy
import numpy as np
from jinja2 import Environment, FileSystemLoader, select_autoescape
import datetime


def render_page(
        data,
        data_id=1,
        write=True,
        template_folder=None,
        template_fn=None,
        output_dir=None,
        cmd_args=None,
        *args,
        **kwargs,
):
    assert (cmd_args is not None ) or not (template_folder is None or template_fn is None or output_dir is None)
    if cmd_args is not None:
        template_folder = cmd_args.template_folder
        template_fn = cmd_args.template
        output_dir = cmd_args.out_dir

    env = Environment(
        loader=FileSystemLoader(template_folder),
        autoescape=select_autoescape(['html', 'xml'])
    )

    template = env.get_template(template_fn)
    html = template.render(
        data=data,
        doc_id=data_id,
        do_mturk=True,
        start_time=str(datetime.datetime.now()),
        **kwargs
    )
    task_label = "task-{}".format(data_id)
    if write:
        outfile_path = os.path.join(output_dir, '{}.html'.format(task_label))
        with open(outfile_path, "w") as file:
            file.write(html)
    return {
        'label': task_label,
        'html': html
    }


def get_worker_requirements(num_hits_threshold=80, country_list=['US'], percent_assignments_approved=90):
    """
    Formulate a worker requirements list.

    :param num_hits_threshold: This specifies how many HITs the worker needs to have completed.
        type: either an int (80) or `False` to not use this.
    :param country_list: Specifies which countries the worker must be based in.
        type: either a list of strings or `False` not to use this.
    :param percent_assignments_approved: what percent assignments are approved from these workers.
        type: either an int or `False` not to use this.
    :return:
    """
    worker_requirements = []
    if num_hits_threshold != False:
        ### number of hits approved
        worker_requirements.append({
            'QualificationTypeId': '000000000000000000L0',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [80],
        })

    if country_list != False:
        locale_vals = list(map(lambda c: {"Country": c}, country_list))
        ## worker locale
        worker_requirements.append({
            'QualificationTypeId': '00000000000000000071',
            'Comparator': 'EqualTo',
            'LocaleValues': locale_vals,
            'RequiredToPreview': True,
        })

    if percent_assignments_approved != False:
        ## percent assignments approved
        worker_requirements.append({
            'QualificationTypeId': '000000000000000000L0',
            'Comparator': 'GreaterThanOrEqualTo',
            'IntegerValues': [percent_assignments_approved],
        })

    return worker_requirements


def launch_HIT(
        mturk_handler,
        question_html,
        worker_requirements=None,
        description=None,
        title=None,
        keywords=None,
        reward=None,
        max_assignments=None,
        HIT_lifetime=None,
        assignment_duration=None,
        assignment_autoapproval=None,

):
    """
    Launches the HITs.

    :param mturk_handler: Instance of `mturk_handler.MTurkHandler`.
    :param question_html: HTML for the question we want to launch.
    :param args: contains CL args that have the following fields:
        :param title: Title (to appear in the MTurk task list).
        :param description: Description (to appear in the MTurk task list).
        :param reward: the payment.
        :param max_assignments: # of times the HIT can be completed.
        :param HIT_lifetime: how long the HIT will remain on the MTurk task list.
        :param assignment_duration: how long a worker can take with the task.
        :param assignment_autoapproval: how before automatically approving the assignment.
    :param worker_requirements: Worker requirements; if the user wants to specify requirements, otherwise they will
                                be instantialized from `get_worker_requirements` with default values.
    :return:
    """
    worker_requirements = worker_requirements or get_worker_requirements()
    new_hit = mturk_handler.client.create_hit(
        Title=title or '',
        Description=description or '',
        Keywords=keywords or 'text, sorting, highlighting',
        Reward=reward or '0.30',
        MaxAssignments=max_assignments or 3,
        LifetimeInSeconds=HIT_lifetime or 172800,
        AssignmentDurationInSeconds=assignment_duration or 6000,
        AutoApprovalDelayInSeconds=assignment_autoapproval or 28800,
        Question=question_html,
        QualificationRequirements=worker_requirements
    )
    hit_id = new_hit['HIT']['HITId']
    return new_hit, hit_id


def get_parser():
    parser = argparse.ArgumentParser()
    # Required parameters
    ### Seed
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        required=False,
        help="The random seed.",
    )
    ### I/O
    parser.add_argument(
        "--data_jsonl_files",
        default=None,
        type=str,
        required=True,
        nargs="+",
        help="The input jsonl files with general samples.",
    )
    parser.add_argument(
        "--qual_jsonl_files",
        default=None,
        type=str,
        required=True,
        nargs="+",
        help="The input jsonl files with qualification samples.",
    )
    parser.add_argument(
        "--out_dir",
        default="html_files",
        type=str,
        required=False,
        help="The output folder to store the html files.",
    )
    parser.add_argument(
        "--img_root",
        default=None,
        type=str,
        required=False,
        help="The image root folder.",
    )
    parser.add_argument(
        '--template_folder',
        default='templates',
        type=str,
        help='Folder containing the template files for the Jinja environment.'
    )
    ###
    ### HIT details
    parser.add_argument(
        "--num_items_per_sample",
        default=5,
        type=int,
        required=False,
        help="The number of event items in each sample.",
    )
    parser.add_argument(
        "--num_samples_per_hit",
        default=5,
        type=int,
        required=False,
        help="The number of data points per HIT (page).",
    )
    parser.add_argument(
        "--num_qual_per_hit",
        default=1,
        type=int,
        required=False,
        help="The number of qualification data points per HIT (page).",
    )
    parser.add_argument(
        "--num_total_hits",
        default=1000,
        type=int,
        required=False,
        help="The number of total HITs (pages).",
    )
    parser.add_argument(
        "--mix_task", action="store_true",
        help="If mixing the tasks in the json file in one HIT."
    )
    parser.add_argument(
        "--mix_modality", action="store_true",
        help="If mixing the modalities in one HIT."
    )
    parser.add_argument(
        "--max_word_per_sentence",
        default=None,
        type=int,
        required=False,
        help="The maximum number of words per each sentence text.",
    )
    # MTurk Handling
    parser.add_argument(
        '--launch_HITs',
        action='store_true',
        help='Whether to launch the HITs straight to '
    )
    parser.add_argument(
        '--mturk_env',
        type=str,
        default='sandbox',
        required=False,
        help="Where to launch MTurk tasks. Options \in {'sandbox', 'production'}."
    )
    parser.add_argument('--title', type=str, default=None, required=False,
                        help='For MTurk, title of the task in HIT list page.')
    parser.add_argument('--description', type=str, default=None, required=False,
                        help='For MTurk, description of the task in HIT list page.')
    parser.add_argument('--keywords', type=str, default=None, required=False,
                        help='For MTurk, keywords for the task in HIT list page.')
    parser.add_argument('--reward', type=str, default=None, required=False, help='For MTurk, payment for turkers.')
    parser.add_argument('--max_assignments', type=str, default=None, required=False,
                        help='For MTurk, number of times each HIT can be completed.')
    parser.add_argument('--HIT_lifetime', type=str, default=None, required=False,
                        help='For MTurk, how long a HIT will continue to be displayed.')
    parser.add_argument('--assignment_duration', type=str, default=None, required=False,
                        help='For MTurk, how long a worker can take with a HIT.')
    parser.add_argument('--assignment_autoapproval', type=str, default=None, required=False,
                        help='For MTurk, how long before automatically approving the work.')
    return parser


def main(args=None):
    # set seed.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # set up OS for reading/writing files.
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    else:
        overwrite = input("Folder: {} exists, overwrite (y/n)? ".format(args.out_dir))
        if overwrite == "y":
            shutil.rmtree(args.out_dir)
            os.makedirs(args.out_dir)
        else:
            exit(-1)

    # Check the number of input files.
    # For mixed tasks, there are several JSON files used whereas for unimodal tasks there's only one.
    if args.mix_task:
        assert len(args.data_jsonl_files) > 1, ("Mixing task but only have one"
            " type of jsonl file from a single task.")
        assert len(args.qual_jsonl_files) > 1, ("Mixing task but only have one"
            " type of jsonl file from a single task.")
    else:
        assert len(args.data_jsonl_files) == 1, ("Not mixing task but have more"
            " than one type of task json files.")
        assert len(args.qual_jsonl_files) == 1, ("Not mixing task but have more"
            " than one type of task json files.")

    # Read in input files.
    general_samples = []
    qualifi_samples = []
    for data_jsonl_file in args.data_jsonl_files:
        general_samples += read_jsonl_file(data_jsonl_file)
    for qual_jsonl_file in args.qual_jsonl_files:
        qualifi_samples += read_jsonl_file(qual_jsonl_file)

    # Calcuate number of general samples and check if we have enough input.
    args.num_gene_per_hit = args.num_samples_per_hit - args.num_qual_per_hit
    if len(general_samples) < args.num_gene_per_hit * args.num_total_hits:
        raise ValueError("Not enough general samples!")
    if len(qualifi_samples) < args.num_qual_per_hit * args.num_total_hits:
        raise ValueError("Not enough qualification samples!")

    # Shuffle the whole sample sets if mixing is required.
    if args.mix_task:
        random.shuffle(general_samples)
        random.shuffle(qualifi_samples)

    # TODO(telinwu): Add qualification samples to general samples for every
    # N data points. Default N = 5.
    HIT_htmls = construct_hits(general_samples, qualifi_samples, args=args)
    if args.launch_HITs:
        from mturk_handler import MTurkHandler
        ## launch HITs
        mturk_client = MTurkHandler(environment=args.mturk_env)
        launched_hits = {}
        for HIT_html in HIT_htmls:
            hit, hit_id = launch_HIT(mturk_client, HIT_html['html'], args)
            launched_hits[hit_id] = hit
        return launched_hits
    else:
        return HIT_htmls



if __name__ == "__main__":
    # get CLI.
    parser = get_parser()
    args = parser.parse_args()
    ##
    output = main(args=args)
    # if args.


