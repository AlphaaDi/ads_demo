def get_newest_user_ready_video(sql_handler, client_id):
    query_result = sql_handler.get_newest_user_ready_video(client_id)
    if len(query_result) == 0:
        video_id = 'woman'
    else:
        video_id = query_result[0][0]
    return video_id


def get_newest_user_ready_logo(sql_handler, client_id):
    query_result = sql_handler.get_newest_user_ready_logo(client_id)
    if len(query_result) == 0:
        image_id = 'logo_example'
    else:
        image_id = query_result[0][0]
    return image_id

def check_status(sql_handler, file_id):
    query_result = sql_handler.check_status(file_id)
    if len(query_result) == 0:
        status = 'Bad_request'
    else:
        status = query_result[0][0]
    return status