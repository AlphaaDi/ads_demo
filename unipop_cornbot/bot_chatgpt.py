import os
import io
import time
import yaml
import argparse
import sqlite3
import hashlib
import urllib.request
from pathlib import Path
import asyncio

from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaVideo
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ConversationHandler, CallbackContext

from sql_handler import SQLHandler
from utils import *

parser = argparse.ArgumentParser(description='',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--token', help='bot api token', type=str)
parser.add_argument('--config_path', default='config.yaml', type=str)

# Define the states for the conversation handler
SELECTING_ACTION, UPLOADING_LOGO, UPLOADING_VIDEO = range(3)


sql_handler = None
blob_storage_path = None
CONFIG = None

# Define the menu and its buttons
menu = [
    [InlineKeyboardButton("Upload logo and get videos", callback_data='upload_logo')],
    [InlineKeyboardButton("Upload video", callback_data='upload_video')],
    [InlineKeyboardButton("Show demo videos", callback_data='get_demo_video')]
]

reply_markup = InlineKeyboardMarkup(menu)


# Define the handlers for the conversation handler
async def start(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Welcome to the Media Uploader Bot!\nPlease select an option:", reply_markup=reply_markup)
    return SELECTING_ACTION

async def helping(update: Update, context: CallbackContext) -> int:
    await update.message.reply_text("Please select an option:", reply_markup=reply_markup)
    return SELECTING_ACTION

async def select_action(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    context.user_data['action'] = query.data
    
    if context.user_data['action'] == 'upload_logo':
        await query.edit_message_text("Please upload a logo and we process our videos for you.")
        return UPLOADING_LOGO
    elif context.user_data['action'] == 'upload_video':
        await query.edit_message_text("Please upload a video and we generate mesh for it")
        return UPLOADING_VIDEO
    elif context.user_data['action'] == 'get_demo_video':        
        for video_path in CONFIG['demo_videos']:
            await context.bot.send_video(chat_id=update.effective_chat.id, video=video_path)

        context.user_data.clear()
        return ConversationHandler.END


async def process_file(file_object):
    file = await file_object.get_file()
    file_id = file.file_id

    file_extension = file.file_path.split('.')[-1]
    file_name = f"{file_id}.{file_extension}"

    file_path = str(blob_storage_path / file_name)
    urllib.request.urlretrieve(file.file_path, file_path)
    return file_id, file_path
    

async def handle_video_to_process(update: Update, context: CallbackContext) -> int:
    if update.message.video is not None:
        file_object = update.message.video
    else:
        file_object = update.message.document

    
    file_id, file_path = await process_file(file_object)
    file_type = 'video'
    
    # Insert the file name or hash into the database
    client_id = update.effective_chat.id
    sql_handler.insert_file(
        file_id, client_id, file_type, file_path, 
        status=CONFIG['statuses_dict']['need_process']
    )
    # Send a response to the user
    await update.message.reply_text("Video stored into query and will be processed!")
    await update.message.reply_text("You can check status by file id:")
    await update.message.reply_text(f"/check_video {file_id}")
    
    # End the conversation
    context.user_data.clear()
    return ConversationHandler.END


async def send_placement_result(update, context, video_id, logo_id):
    result_video_path = sql_handler.get_logo_placement_result(video_id, logo_id)[0][0]

    await update.message.reply_text(f'Process result:')
    await context.bot.send_video(chat_id=update.effective_chat.id, video=result_video_path)

async def process_and_send_result(update, context, video_id, logo_id):
    # Perform the backend operation and wait for the result
    while True:
        query_result = sql_handler.check_placement_task(video_id, logo_id)
        if len(query_result) == 0:
            break
        query_count = sql_handler.get_placement_task_queue_num(video_id, logo_id)
        await update.message.reply_text(f'Task place in queue {query_count}')
        time.sleep(10)

    await send_placement_result(update, context, video_id, logo_id)


async def handle_logo(update: Update, context: CallbackContext) -> int:
    if len(update.message.photo) != 0:
        file_object = update.message.photo[0]
        await update.message.reply_text("Load the png file logo for transparent square!")
    else:
        file_object = update.message.document
    
    file_id, file_path = await process_file(file_object)
    file_type = 'image'
    # Insert the file name or hash into the database
    client_id = update.effective_chat.id
    sql_handler.insert_file(file_id, client_id, file_type, file_path)
    # Send a response to the user
    await update.message.reply_text("Logo stored into query and will be processed !")
    await update.message.reply_text("This logo is your current logo to use...")
    
    client_id = update.effective_chat.id
    video_id = get_newest_user_ready_video(sql_handler, client_id)

    sql_handler.insert_logo_place_task(video_id, file_id)
    asyncio.create_task(
        process_and_send_result(update, context, video_id, file_id)
    )
    context.user_data.clear()
    return ConversationHandler.END


async def cancel(update: Update, context: CallbackContext) -> int:
    update.message.reply_text("Conversation cancelled.")
    context.user_data.clear()
    return ConversationHandler.END


async def do_placement_with_check(update, context, video_id, logo_id):
    status = check_status(sql_handler, video_id)

    if status != CONFIG['statuses_dict']['ready']:
        await update.message.reply_text(f"Your video status is {status}, please process or wait video first :)")
        return

    placement_result = sql_handler.get_logo_placement_result(video_id, logo_id)
    if placement_result:
        await send_placement_result(update, context, video_id, logo_id)
    else:
        sql_handler.insert_logo_place_task(video_id, logo_id)
        asyncio.create_task(
            process_and_send_result(update, context, video_id, logo_id)
        )


async def check_video(update: Update, context: CallbackContext):
    video_id = update.message.text.split(' ')[-1]
    logo_id = CONFIG['logo_example']

    do_placement_with_check(update, context, video_id, logo_id)


async def process_reply_video(update: Update, context: CallbackContext):
    if update.message.reply_to_message:
        if update.message.reply_to_message.video is not None:
            original_video = update.message.reply_to_message.video
        elif update.message.reply_to_message.document is not None:
            original_video = update.message.reply_to_message.document
        else:
            await update.message.reply_text("Please reply command on video message")
            return       
    else:
        await update.message.reply_text("Please reply command on message")
        return

    client_id = update.effective_chat.id
    logo_id = get_newest_user_ready_logo(sql_handler, client_id)

    file = await original_video.get_file()
    video_id = file.file_id

    await do_placement_with_check(update, context, video_id, logo_id)


async def process_reply_msg(update: Update, context: CallbackContext):
    if update.message.reply_to_message:
        if update.message.reply_to_message.image is not None:
            original_image = update.message.reply_to_message.image
        elif update.message.reply_to_message.document is not None:
            original_image = update.message.reply_to_message.document
        else:
            await update.message.reply_text("Please reply command on image message")
            return       
    else:
        await update.message.reply_text("Please reply command on message")
        return

    file = await original_image.get_file()
    logo_id = file.file_id

    client_id = update.effective_chat.id
    video_id = get_newest_user_ready_video(sql_handler, client_id)

    await do_placement_with_check(update, context, video_id, logo_id)


# Define the main function to run the bot
def main():
    args = parser.parse_args()
    config = yaml.full_load(open(args.config_path, 'r'))
    
    global sql_handler
    sql_handler = SQLHandler(config['media_files_db'])
    
    global blob_storage_path
    blob_storage_path = Path(config['blob_storage_path'])
    
    global CONFIG
    CONFIG = config
    
    application = ApplicationBuilder().token(args.token).build()
    
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start),
            CommandHandler('help', helping)
        ],
        states={
            SELECTING_ACTION: [CallbackQueryHandler(select_action)],
            UPLOADING_LOGO: [MessageHandler(
                (filters.PHOTO | filters.Document.IMAGE) & ~filters.COMMAND, handle_logo
            )],
            UPLOADING_VIDEO: [MessageHandler(
                (filters.VIDEO | filters.Document.VIDEO) & ~filters.COMMAND, handle_video_to_process
            )],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    application.add_handler(conv_handler)
    
    help_handler = CommandHandler('help', helping)
    application.add_handler(help_handler)
    
    check_video_handler = CommandHandler('check_video', check_video)
    application.add_handler(check_video_handler)

    process_reply_video_handler = CommandHandler('process_reply_video', process_reply_video)
    application.add_handler(process_reply_video_handler)

    process_reply_image_handler = CommandHandler('process_reply_message', process_reply_msg)
    application.add_handler(process_reply_image_handler)
    
    application.run_polling()


main()
