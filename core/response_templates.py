RESPONSE_TEMPLATES = {
    "initial": "👋 Hi {user_name}! I'm SplitBot. Send me a bill (image, text, or voice) to get started!",
    "completed_already": "This split is already completed, {user_name}! Here was the summary:\n{final_summary}\nType /start for a new one.",
    "ask_for_bill_again": "I'm ready to help, {user_name}! Please send the bill details or a picture.",
    "bill_extraction_failed": "Sorry {user_name}, I couldn't extract the bill details. Please try again or describe it differently.",
    "bill_summary": "📋 Bill Summary for {user_name}:\n🏪 Restaurant: {restaurant}\n💰 Total: {total}\n📝 Items Preview:\n{items_preview}\n\nHow would you like to split this?",
    "split_confirmation": "✅ Split Confirmation for {user_name}:\n💰 Total: {total}\n{person_breakdown}\n\nIs this correct? (Yes/No/Adjust)",
    "split_calculation_failed": "I had trouble understanding how to split that, {user_name}. Could you phrase it differently?",
    "split_calculation_needs_clarification": "I need a bit more information to calculate that split, {user_name}. Details: {details}",
    "completed": "🎉 Split completed, {user_name}! Here's your final summary:\n{final_summary}\n\nTo start a new split, type /start",
    "ask_for_adjustment_details": "Okay {user_name}, what changes would you like to make?",
    "confirmation_not_understood": "Sorry {user_name}, I didn't catch that. Please say 'Yes' to confirm, or 'No'/'Adjust' to make changes.",
    "adjustment_re_prompt_split": "Got it, {user_name}. Let's redefine the split. How should we divide the bill now based on the items I found?",
    "error_generic": "Sorry {user_name}, an unexpected error occurred. Please try /start again.",
    "error_in_state": "Sorry {user_name}, I encountered an issue in state: {state}. Please try /start again."
} 