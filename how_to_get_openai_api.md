# OpenAI API Key Usage 

## Overview
This README document provides step-by-step instructions on how to obtain and use the OpenAI API key. The API key will grant you access to interact with the models like GPT-3, GPT-4, etc for various purposes.

## Prerequisites
To use the OpenAI API, you'll need to have:
- An OpenAI account
- Python installed on your local machine (Python 3.6 or later)
- OpenAI's Python client library

## Steps
### Step 1: Creating an OpenAI Account
If you haven't created an OpenAI account, you'll need to create one.
1. Visit https://beta.openai.com/signup/
2. Fill in your details (Full Name, Email, Password)
3. Click "Sign up"

### Step 2: Get your API key
After successfully creating and verifying your account:

1. Log in to your OpenAI account
2. Navigate to the API section by clicking on your username at the top right of the dashboard, then select "API Keys" from the dropdown menu
3. Click the "Create API Key" button
4. Label your key for better identification (optional), and click "Create"
5. Your new API key will be created. Ensure to copy it and store it securely, you won't be able to see it again for security reasons


# AFTER OBTAINING THE API KEY, YOU CAN GO BACK AND CONTINUE TO RUN [OUR STARTING KIT](https://colab.research.google.com/drive/1lvnIbdQcXx8vzV4KL62KfWEsQedNArxd?usp=sharing) 

Remember, your API key is confidential. Avoid sharing it, as it provides full access to your OpenAI account. At first you will be given a certain amount of free quota for using it, exceeding this threshold you will have to pay using your own credit card.

### Step 3: Installing OpenAI's Python client library
Open a terminal window and install the OpenAI Python client using pip:

```bash
pip install openai
```

### Step 4: Using the API key in your Python code
Now you can use the API key in your Python code. Store the key in an environment variable for security reasons, rather than hard coding it into your script.

For UNIX and MacOS:

```bash
export OPENAI_API_KEY='your-api-key'
```

For Windows:

```cmd
set OPENAI_API_KEY=your-api-key
```

Then in your Python script, you can try something simple like asking who won the world series in 2020:

```python 
import openai

openai.api_key = os.getenv('OPENAI_API_KEY')

openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"}
    ]
)

print(response.choices[0].text.strip())
```

Remember to replace `your-api-key` with your actual API key.

That's it! You should now be able to use the OpenAI APIs for your applications. For further information on how to use the API, you can check the official OpenAI API [documentation](https://platform.openai.com/docs/introduction). Enjoy coding!
