import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes

from telegram import (
    KeyboardButton,
    KeyboardButtonPollType,
    Poll,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_markup = ReplyKeyboardMarkup(True, True)
    user_markup.row('Upload logotype 📷')
    user_markup.row('Upload video to process 🎞')
    user_markup.row('Get demo 🏞')
    user_markup.row('Help ❓')
    message = (
        "Hello there! Nice to meet you! \n"
        "I could show you how to create virtual product placement logo: \n"
        "1) We can show our demo \n"
        "2) It could be your examples on our videos \n"
        "3) Or we could process your video \n"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=message, reply_markup=user_markup)


# @bot.message_handler(regexp="👈 Main Menu")
# def main_menu(m):


async def upload_logo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.bot.register_next_step_handler(message, get_logo)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Please upload your logo")

async def get_logo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(update.effective_chat.id)
    photo = message.photo;
    print(message.photo[0].file_id)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Got you")

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")


application = ApplicationBuilder().token('6067579365:AAGjXnr2R4x0WextBkkKcRXnBoklKRzHWbw').build()

start_handler = CommandHandler('start', start)
application.add_handler(start_handler)

# upload_logo_handler = CommandHandler('upload_logo', upload_logo)
# application.add_handler(upload_logo_handler)

unknown_handler = MessageHandler(filters.COMMAND, unknown)
application.add_handler(unknown_handler)

application.run_polling()
