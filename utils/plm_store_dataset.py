import argparse
from utils.vocab import tokenize
from transformers import BertTokenizerFast
import ujson as json
from tqdm import tqdm
import pickle
import copy
from pprint import pprint

def create_answer_mapping(annotations, ans2cat):
    """Returns mapping from question_id to answer.

    Only returns those mappings that map to one of the answers in ans2cat.

    Args:
        annotations: VQA annotations file.
        ans2cat: Map from answers to answer categories that we care about.

    Returns:
        answers: Mapping from question ids to answers.
        image_ids: Set of image ids.
    """
    answers = {}
    image_ids = set()
    for q in annotations['annotations']:
        question_id = q['question_id']
        answer = q['multiple_choice_answer']
        if answer in ans2cat:
            answers[question_id] = answer
            image_ids.add(q['image_id'])
    return answers, image_ids


def store_dataset(image_dir, train_questions, val_questions, train_annotations, val_annotations, ans2cat, cat2name, output,
                 im_size=224, max_q_length=20, max_a_length=4,
                 with_answers=False, train_or_val="train"):

    tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    special_tokens = ["key_"+cat for cat in cat2name]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    dataset_object_shape_base = {
        "tokenized_question": {},
        "english_question": [],
        "tokenized_answer": {},
        "english_answer": [],
        "tokenized_category": {},
        "english_category": [],
        "tokenized_cat_ans": {},
        "english_cat_ans": [],
        "tokenized_cat_ques": {},
        "english_cat_ques": [],
        "image_id": "",
        # image_features: []
    }
    train_dataset={"data": []}
    val_dataset={"data": []}

    # Load the data.
    with open(train_annotations) as f:
        train_annos = json.load(f)
    with open(train_questions) as f:
        train_questions = json.load(f)
    with open(val_annotations) as f:
        val_annos = json.load(f)
    with open(val_questions) as f:
        val_questions = json.load(f)

    train_qid2ans, image_ids = create_answer_mapping(train_annos, ans2cat)
    val_qid2ans, image_ids = create_answer_mapping(val_annos, ans2cat)

    def build_data_object(question, answer, category, image_id):
        dataset_object_shape = copy.deepcopy(dataset_object_shape_base)

        cat_ans = category, answer
        cat_ques = category, question
        fields = [question, answer, category, cat_ans, cat_ques]
        field_strings = ["question", "answer", "category", "cat_ans", "cat_ques"]
        max_lens = [20,6,3,9,22]

        for i, field in enumerate(fields):
            if field_strings[i] == "cat_ans" or field_strings[i] == "cat_ques":
                tokenized = tokenizer(*field, add_special_tokens=True, return_tensors="pt", padding="max_length", truncation=True, max_length=max_lens[i])
            else:
                tokenized = tokenizer(field, add_special_tokens=True, return_tensors="pt", padding="max_length", truncation=True, max_length=max_lens[i])
            for k, v in dict(tokenized).items():
                tokenized[k] = v.numpy().tolist()
            
            dataset_object_shape["tokenized_"+field_strings[i]] = dict(tokenized)
            dataset_object_shape["english_"+field_strings[i]] = tokenizer.convert_ids_to_tokens(tokenized.input_ids[0])

        dataset_object_shape["image_id"] = image_id

        return dataset_object_shape



    for i, entry in enumerate(tqdm(train_questions["questions"])):

        question_id = entry['question_id']
        if question_id not in train_qid2ans:
            continue

        answer_string = train_qid2ans[question_id]
        category_id = int(ans2cat[answer_string])
        category_string = "key_"+cat2name[category_id]

        dataset_object_shape = build_data_object(entry["question"], answer_string, category_string, entry["image_id"])

        train_dataset["data"].append(dict(dataset_object_shape))

        # if i == 10: break


    for i, entry in enumerate(tqdm(val_questions["questions"])):

        question_id = entry['question_id']
        if question_id not in val_qid2ans:
            continue

        answer_string = val_qid2ans[question_id]
        category_id = int(ans2cat[answer_string])
        category_string = "key_"+cat2name[category_id]

        dataset_object_shape = build_data_object(entry["question"], answer_string, category_string, entry["image_id"])

        val_dataset["data"].append(dict(dataset_object_shape))

        # if i == 10: break

    json.dump(dict(train_dataset), open(output+"train_processed_dataset.json", "w"))
    json.dump(dict(val_dataset), open(output+"val_processed_dataset.json", "w"))
    pickle.dump(tokenizer, open(output+"tokenizer.pkl", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs.
    parser.add_argument('--image-dir', type=str, default='data/vqa/train2014',
                        help='directory for resized images')
    parser.add_argument('--train_questions', type=str,
                        default='data/vqa/v2_OpenEnded_mscoco_'
                        'train2014_questions.json',
                        help='Path for train annotation file.')
    parser.add_argument('--train_annotations', type=str,
                        default='data/vqa/v2_mscoco_'
                        'train2014_annotations.json',
                        help='Path for train annotation file.')
    parser.add_argument('--val_questions', type=str,
                        default='data/vqa/v2_OpenEnded_mscoco_'
                        'val2014_questions.json',
                        help='Path for train annotation file.')
    parser.add_argument('--val_annotations', type=str,
                        default='data/vqa/v2_mscoco_'
                        'val2014_annotations.json',
                        help='Path for train annotation file.')

    parser.add_argument('--cat2ans', type=str,
                        default='data/vqa/iq_dataset.json',
                        help='Path for the answer types.')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for saving vocabulary wrapper.')

    # Outputs.
    parser.add_argument('--output_dir', type=str,
                        default='data/processed/',
                        help='directory for processed files.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=224,
                        help='Size of images.')


    # Train or Val?
    parser.add_argument('--val', type=bool, default=False, help="whether we're working iwth the validation set or not")
    args = parser.parse_args()

    ans2cat = {}
    with open(args.cat2ans) as f:
        cat2ans = json.load(f)
    cats = sorted(cat2ans.keys())
    with open(args.cat2name, 'w') as f:
        json.dump(cats, f)
    for cat in cat2ans:
        for ans in cat2ans[cat]:
            ans2cat[ans] = cats.index(cat)

    store_dataset(args.image_dir, args.train_questions,args.val_questions, args.train_annotations, args.val_annotations,
                 ans2cat, cats, args.output_dir, im_size=args.im_size)
    print(('Wrote dataset to %s' % args.output_dir))
