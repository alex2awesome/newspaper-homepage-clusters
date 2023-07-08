import pandas as pd
import xmltodict
import boto3
from tqdm.auto import tqdm
import json
import datetime, time
import os


ACCESS_KEY = os.environ.get('JONMAY_AWS_ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.environ.get('JONMAY_AWS_SECRET_ACCESS_KEY')
# ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY_ID')
# SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

class MTurkHandler():
    def __init__(self, environment='production'):
        self._mturk_client = {}
        self.client = self.get_client(env=environment)

    def get_client(self, env='production'):
        """
        Get MTurk client.

        parameters:
            * env: `production` or `sandbox`

        """
        if env in self._mturk_client:
            self.client = self._mturk_client[env]
            return self._mturk_client[env]

        ##### mturk client
        region = 'us-east-1'
        environments = {
          "production": {
            "endpoint": "https://mturk-requester.%s.amazonaws.com" % region,
            "preview": "https://www.mturk.com/mturk/preview"
          },
          "sandbox": {
            "endpoint":
                  "https://mturk-requester-sandbox.%s.amazonaws.com" % region,
            "preview": "https://workersandbox.mturk.com/mturk/preview"
          },
        }

        session = boto3.Session(profile_name='mturk')
        self._mturk_client[env] = session.client(
            service_name='mturk',
            region_name=region,
            endpoint_url=environments[env]['endpoint'],
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_ACCESS_KEY,
        )
        self.client = self._mturk_client[env]
        return self._mturk_client[env]

    def iterate_hits(self, retrieve_hit_func):
        all_hits = []
        hits = retrieve_hit_func()
        all_hits += hits['HITs']
        while 'NextToken' in hits:
            next_tok = hits['NextToken']
            hits = retrieve_hit_func(NextToken=next_tok)
            all_hits += hits['HITs']
        all_hit_ids = list(map(lambda x: x['HITId'], all_hits))
        return all_hit_ids, all_hits

    def get_all_hits(self):
        return self.iterate_hits(self.client.list_hits)

    def get_all_reviewable_hits(self):
        return self.iterate_hits(self.client.list_reviewable_hits)


    def remove_hits(self, hit_or_hit_list):
        if isinstance(hit_or_hit_list, str):
            hit_list = [hit_or_hit_list]
        else:
            hit_list = hit_or_hit_list

        for hit_id in hit_list:
            # Get HIT status
            status = self.client.get_hit(HITId=hit_id)['HIT']['HITStatus']

            # If HIT is active then set it to expire immediately
            if status == 'Assignable':
                response = self.client.update_expiration_for_hit(
                    HITId=hit_id,
                    ExpireAt=datetime.datetime.now()
                )

            time.sleep(1)
            self.client.delete_hit(HITId=hit_id)

    def get_answer_df_for_hit_list(self, hit_list):
        answer_dfs = []
        for hit_id in tqdm(hit_list):
            ##
            assignmentsList = self.client.list_assignments_for_hit(
                HITId=hit_id,
                AssignmentStatuses=['Submitted', 'Approved'],
                MaxResults=10
            )
            assignments = assignmentsList['Assignments']
            assignments_submitted_count = len(assignments)
            for assignment in assignments:
                # Retreive the attributes for each Assignment
                answer_dict = xmltodict.parse(assignment['Answer'])
                print(type(answer_dict['QuestionFormAnswers']['Answer']), len(answer_dict['QuestionFormAnswers']['Answer']))
                if type(answer_dict['QuestionFormAnswers']['Answer']) != list:
                    print(answer_dict['QuestionFormAnswers']['Answer'].keys())
                    answer = json.loads(answer_dict['QuestionFormAnswers']['Answer']['FreeText'])
                else:
                    print(answer_dict['QuestionFormAnswers']['Answer'])
                    answer = json.loads(answer_dict['QuestionFormAnswers']['Answer'][1]['FreeText'])
                answer_df = pd.DataFrame(answer)
                answer_df['worker_id'] = assignment['WorkerId']
                answer_df['assignment_id'] = assignment['AssignmentId']
                answer_df['hit_id'] = assignment['HITId']
                answer_df['time_delta'] = assignment['SubmitTime'] - assignment['AcceptTime']
                answer_dfs.append(answer_df)
        return pd.concat(answer_dfs)

    def get_hit_status(self, hit_list):
        """
        Get's the status for all HITs of a given list.

        :param hit_list:
        :return:
        """
        output = []
        for hit_id in hit_list:
            hit = self.client.get_hit(HITId=hit_id)
            output.append(hit['HIT'])
        return output

    def reject_assignment(self, assignment_list, comment=None):
        if comment is None:
            comment = 'You didn\'t select anything!'

        failed = 0
        for assignment_id in tqdm(assignment_list):
            try:
                response = self.client.reject_assignment(
                    AssignmentId=assignment_id,
                    RequesterFeedback=comment
                )
            except:
                failed += 1

    def block_workers(self, worker_list, comment=None):
        if comment is None:
            comment = 'You did not fill out any questions on our form.'
        for worker_id in worker_list:
            response = self.client.create_worker_block(
                WorkerId=worker_id,
                Reason=comment
            )


    def create_qualification(self, qual_name, qual_keywords, qual_description):
        response = self.client.create_qualification_type(
            Name=qual_name,
            Keywords=qual_keywords,
            Description=qual_description,
            QualificationTypeStatus='Active',
        )
        return response


    def give_qualification_to_workers(self, worker_ids, qualification_id):
        """

        """

        for worker_id in worker_ids:
            response = self.client.associate_qualification_with_worker(
                QualificationTypeId=qualification_id,
                WorkerId=worker_id,
                IntegerValue=100,
                SendNotification=True,
            )
