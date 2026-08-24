import os
import logging

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

from agent import app_graph

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    chat_id = update.effective_chat.id

    status_msg = await context.bot.send_message(chat_id=chat_id, text="Thinking...")

    pdf_path = None
    try:
        result = app_graph.invoke({"question": question, "retry_count": 0})
        pdf_path = result.get("pdf_path")

        if not pdf_path or not os.path.exists(pdf_path):
            await context.bot.edit_message_text(
                chat_id=chat_id, message_id=status_msg.message_id,
                text="Sorry, I couldn't generate a report for that question."
            )
            return

        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=status_msg.message_id,
            text="Done - here's your report."
        )
        with open(pdf_path, "rb") as f:
            await context.bot.send_document(chat_id=chat_id, document=f, filename="report.pdf")

    except Exception as e:
        logger.exception("Error handling message")
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=status_msg.message_id,
            text=f"Something went wrong: {e}"
        )
    finally:
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)


def main():
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set. Check your .env file.")

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot starting...")
    application.run_polling()


if __name__ == "__main__":
    main()
