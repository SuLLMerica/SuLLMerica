import ujson as json
import re

def generate_questions_training(path):
    with open(path) as json_file:
        docs = json.load(json_file)
    questions = []
    for key, value in docs.items():
        question = {}
        question['code'] = key.split(' ')[1]
        question['question'] = value['question']
        
        # Modificação para que options seja um dicionário
        options = {}
        for code, option in value.items():
            if 'option' in code:
                options[code] = option
        question['options'] = options
        
        question['answer'] = int(re.findall(r'[\d]+', value['answer'].split(':')[0])[0])
        questions.append(question)

    return questions


if __name__ == '__main__':
    questions = generate_questions_training("../TeleQnA_teste.json")
    print(questions)

    for question in questions:
            print(f"Code: {question['code']}")
            print(f"Question: {question['question']}")
            print(f"Options: {question['options']}")
            print(f"Answer: {question['answer']}")