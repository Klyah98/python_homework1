import json
import config
import requests
import telebot
from telebot import types
from help_func import json_from_csv_bytes


bot = telebot.TeleBot(config.TOKEN)


@bot.message_handler(commands=['start'])
def welcome(message):
    welcome_text = (
        'Привет, я ML-classification бот!\n'
        'Я помогу тебе обучить модель классификации для твоей задачи :)\n'
        'Чтобы узнать, как я работаю, используй команду /help'
    )
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)

    button1 = types.KeyboardButton('Список доступных моделей')
    button3 = types.KeyboardButton('Список обученных моделей')
    markup.add(button1, button3)

    bot.send_message(message.chat.id, welcome_text, reply_markup=markup)


@bot.message_handler(commands=['help'])
def help(message):
    text = (
        'Данный бот обучает модели бинарной классифцикации на входных данных. Все\n'
        'Доступные команды:\n'
        '\t- Обучить новую модель из списка (список доступен по кнопке "Список доступных моделей").\n'
        'Для этого нужно прикрепить тренировочные данные в формате csv, а в сообщении написать\n'
        'имя модели и название модели из списка, например\n'
        '"Данные для обучения новой модели|random_forest,Random Forest"\n'
        '\t- Переобучить уже существующую модель. Для этого на вход подается csv файл и имя уже существующей\n'
        'модели, например "Данные для переобучения модели|random_forest"\n'
        '\t- Получить предсказания по данным. Для этого нужно прикрепить csv файл с данными\n'
        'и указать название модели, например "Данные для предсказания|random_forest"\n'
        '\t- Удалить существующую модель. Для этого нужно написать "Удалить модель" и имя модели\n'
        'например "Удалить модель|random_forest"'

    )
    bot.send_message(message.chat.id, text)


@bot.message_handler(content_types=['text'])
def process_text_messages(message):
    if message.text == 'Список доступных моделей':
        response = requests.get(config.api_info_endpoint)
        bot.send_message(message.chat.id, response.json())
    if message.text == 'Список обученных моделей':
        response = requests.get(config.api_train_endpoint)
        text = response.json()
        if not text:
            text = 'Нет обученных моделей'
        bot.send_message(message.chat.id, str(text))
    if 'Удалить модель' in message.text:
        model_name = message.text.split(' ')[-1]
        endpoint = config.api_delete_endpoint + '/' + model_name
        requests.delete(endpoint)
        bot.send_message(message.chat.id, 'Модель удалена :)')


@bot.message_handler(content_types=['text', 'document'])
def test_func(message):

    file = bot.get_file(message.document.file_id)
    download_file = bot.download_file(file.file_path)
    json_data = json_from_csv_bytes(download_file)

    body = {}
    if 'Данные для обучения новой модели' in message.caption:
        params = message.caption.split('|')[1]
        body['model_name'] = params.split(',')[0]
        body['model_type'] = params.split(',')[1]
        body['train_data'] = json_data
        response = requests.post(config.api_train_endpoint, json=body)
        metrics = json.loads(response.json())['score']
        bot.send_message(message.chat.id, f'Данные приняты!\nРезультат обучения:\n{metrics}')
    if 'Данные для переобучения модели' in message.caption:
        model_name = message.caption.split('|')[1]
        body['model_name'] = model_name
        body['new_data'] = json_data
        response = requests.put(config.api_train_endpoint, data=body)
        metrics = json.loads(response.json())['score']
        bot.send_message(message.chat.id, f'Данные приняты!\nРезультат обучения:\n{metrics}')
    if 'Данные для предсказания' in message.caption:
        model_name = message.caption.split('|')[1]
        body['model_name'] = model_name
        body['predict_data'] = json_data
        response = requests.post(config.api_predict_endpoint, json=body)
        answers = response.json()
        bot.send_message(message.chat.id, str(answers))


bot.polling(none_stop=True)
