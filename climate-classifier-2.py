import pandas as pd
import openai as ai
from sklearn.metrics import recall_score, precision_score, accuracy_score

ai.api_key = #YOUR API KEY


# Mockup function that classifies the text. Replace with your own.
def classify_paragraph(paragraph):

    response = ai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
        "role": "system",
        "content": "Let's think step by step. You are a labeler. Please identity if the provided paragraph makes a claim associated with climate denialism. Assume these claims are scraped from blogs associated with climate denialism. A paragraph is denialist if the focus is on climate, and if the paragraph uses arguments commonly used in climate change or global warming denial"
        },
        {
        "role": "user",
        "content": "Paragraph: \"It makes you wonder what created all that CO2 millions of years ago.\""
        },
        {
        "role": "assistant",
        "content": "\nStep by Step Analysis:\n\n1) Topic of Focus: The paragraph appears to focus on CO2 (carbon dioxide) and its origins millions of years ago. CO2 is indeed a primary greenhouse gas responsible for global warming.\n\n2) Denialist Claims: One common argument made by climate denialists is pointing to historical or prehistorical periods when the Earth's climate was different as proof that humans are not causing current changes. By asking \"what created all that CO2 millions of years ago,\" the paragraph might be insinuating that natural processes have historically been responsible for CO2 concentrations, and thus today's elevated levels could also be natural.\n\n3) Tone and Implication: The paragraph poses a question rather than directly stating a denialist position. However, the implication may be that if there were high CO2 levels millions of years ago, then current CO2 levels (and associated climate change) might be part of a natural process, not influenced by human activity.\n\n4) Conclusion: The paragraph does appear to hint at a claim associated with climate denialism, specifically suggesting that CO2 levels might have had natural causes in the past and thus implying that current levels could also be natural. While the paragraph is not aggressively denialist, the underlying sentiment does align with arguments commonly used in climate change or global warming denial."
        },
        {
        "role": "user",
        "content": "Paragraph: \"Nuccitellis answer to this unexceptionable and temperately-expressed statement is that sea level rises chiefly through thermal expansion and melting land ice, so it is all our fault. However, it ought to have been obvious even to Nuccitelli thermal expansion and land-ice melt happen whether Man or nature is the cause.\""
        },
        {
        "role": "assistant",
        "content": "Step by Step Analysis:\n\n1) Topic of Focus: The paragraph focuses on sea level rise, a significant consequence of global warming. The causes mentioned include thermal expansion and melting land ice, both of which are associated with global warming.\n\n2) Denialist Claims: The paragraph disputes the claim that humans are responsible for sea level rise due to thermal expansion and land-ice melt. The author suggests that these phenomena can occur naturally, regardless of human influence. This argument aligns with a common climate denialist claim that shifts the blame from human activities to natural processes.\n\n3) Tone and Implication: The tone of the paragraph is dismissive towards the idea that humans are causing sea level rise. The implication is that these processes could be entirely natural, undermining the consensus that human-induced global warming is causing sea level rise.\n\n4) Conclusion: The paragraph makes a claim associated with climate denialism by arguing that sea level rise due to thermal expansion and land-ice melt could be natural processes, not necessarily caused by human activities. This argument is a common tactic used by climate denialists to downplay the impact of human activity on climate change."
        },
        {
        "role": "user",
        "content": f"Paragraph: \"{paragraph}\""
        }
    ],
    temperature=0
    )
    

    extracted_response = ai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {
        "role": "system",
        "content": "You have just been provided analysis from a human labeler. Please convert the analysis into a label. The labeler was asked to identify a claim as climate denialist or not. If the labeler concluded that the claim was denialist, respond \"YES\". If not, respond \"NO\". Please respond with only the label.\n\n\n\n\n\n\n"
        },
        {
        "role": "user",
        "content": "\"Step by Step Analysis:\n\nTopic of Focus: The paragraph appears to focus on CO2 (carbon dioxide) and its origins millions of years ago. CO2 is indeed a primary greenhouse gas responsible for global warming.\n\nDenialist Claims: One common argument made by climate denialists is pointing to historical or prehistorical periods when the Earth's climate was different as proof that humans are not causing current changes. By asking \"what created all that CO2 millions of years ago,\" the paragraph might be insinuating that natural processes have historically been responsible for CO2 concentrations, and thus today's elevated levels could also be natural.\n\nTone and Implication: The paragraph poses a question rather than directly stating a denialist position. However, the implication may be that if there were high CO2 levels millions of years ago, then current CO2 levels (and associated climate change) might be part of a natural process, not influenced by human activity.\n\nConclusion: The paragraph does appear to hint at a claim associated with climate denialism, specifically suggesting that CO2 levels might have had natural causes in the past and thus implying that current levels could also be natural. While the paragraph is not aggressively denialist, the underlying sentiment does align with arguments commonly used in climate change or global warming denial.\""
        },
        {
        "role": "assistant",
        "content": "YES"
        },
        {
        "role": "user",
        "content": "\nAnalysis of the Paragraph:\n\n\"Figure 1. Lake Baikal paleoclimate record from the past 800,000 years. Warmer conditions are towards the right, cooler ones towards the left (from Mackay, 2007).\"\n\nStep-by-Step Analysis:\n\nContent: The paragraph describes a paleoclimate record from Lake Baikal over the past 800,000 years.\n\nNature of Information: The paragraph is presenting data without interpreting it in the context of current climate change trends. It only mentions that the chart or graph shows warmer conditions to the right and cooler ones to the left.\n\nSource Cited: The paragraph cites \"Mackay, 2007\" as the source of this data. Without further information on this source, it cannot be conclusively judged as supporting or opposing mainstream climate science.\n\nTone and Implications: There isn’t a discernible tone or slant that pushes a denialist agenda in this paragraph. It doesn't provide an interpretation of the data or relate it to contemporary climate change discussions.\n\nDenialist Markers: Common markers of climate denialism include:\n\nClaiming that the climate has always changed, so current changes are not unusual.\nArguing that human activities are not responsible for current climate change.\nDownplaying the effects of current climate change.\nSuggesting that there is significant disagreement among scientists about the causes or effects of climate change.\nThis paragraph doesn't seem to display any of these markers.\nConclusion:\nThe paragraph does not make a claim associated with climate denialism. It simply describes a paleoclimate record without adding interpretive or contrasting views related to current climate change discussions."
        },
        {
        "role": "assistant",
        "content": "NO"
        },
        {
        "role": "user",
        "content": f"{response.choices[0].message.content}"
        }
    ],
    temperature=0
    )

    if extracted_response.choices[0].message.content == "YES":
        
        label_raw = ai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": "You are a human labeler who labels climate denial claims.\\nHere are the labels:\\n1.1 - Ice/permafrost/snow cover isn’t melting\\n1.2 - We’re heading into an ice age/global cooling\\n1.3 - Weather is cold/snowing\\n1.4 - Climate hasn’t warmed/changed over the last (few) decade(s)\\n1.6 - Sea level rise is exaggerated/not accelerating\\n1.7 - Extreme weather isn’t increasing/has happened before/isn’t linked to climate change\\n2.1 - It’s natural cycles/variation\\n2.3 - There’s no evidence for greenhouse effect/carbon dioxide driving climate change\\n3.1 - Climate sensitivity is low/negative feedbacks reduce warming\\n3.2 - Species/plants/reefs aren’t showing climate impacts/are benefiting from climate change\\n3.3 - CO2 is beneficial/not a pollutant\\n4.1 - Climate policies (mitigation or adaptation) are harmful\\n4.2 - Climate policies are ineffective/flawed\\n4.4 - Clean energy technology/biofuels won’t work\\n4.5 - People need energy (e.g. from fossil fuels/nuclear)5.2 - Climate science or movement is unreliable, alarmist, or due to corruption\\n\\n \\n5.1 - Climate science is uncertain (methods or models)\\nPlease label the provided claim with the correct label. Ultimately, determine the single best label based on what the claim is mostly about. Let's think step by step."
            },
            {
            "role": "user",
            "content": "It makes you wonder what created all that CO2 millions of years ago."
            },
            {
            "role": "assistant",
            "content": "Alright, let's break down the claim: \"It makes you wonder what created all that CO2 millions of years ago.\"\n\nThis statement is referring to past events and implying that elevated levels of CO2 have occurred naturally in Earth's history without human intervention.\n\nGiven the provided labels:\n\n1.1 - 1.7 mainly pertain to observations about current climate phenomena (e.g., melting ice, extreme weather).\n2.1 - 2.3 are about the causes of climate change, with 2.1 highlighting natural cycles or variations, and 2.3 questioning the evidence for the greenhouse effect or CO2's role in driving climate change.\n3.1 - 3.3 discuss the direct impacts and perceptions of climate change, including the benefits or lack thereof of CO2.\n4.1 - 4.5 are about climate policies and energy choices.\n5.1 - 5.2 discuss the reliability of climate-related science and the motivations of the climate movement.\nGiven this breakdown, the claim appears to align most closely with:\n\n2.1 - It’s natural cycles/variation\n\nThe statement is suggesting that there were natural sources or processes in the Earth's history that led to high levels of CO2, which would fall under the category of natural cycles or variations."
            },
            {
            "role": "user",
            "content": "NBCC/CRA study estimates Waxman-Markey would reduce employment by 2.3 million jobs in 2015"
            },
            {
            "role": "assistant",
            "content": "\nChatGPT\nGiven Claim: \"NBCC/CRA study estimates Waxman-Markey would reduce employment by 2.3 million jobs in 2015\"\n\nThe primary focus of this claim is on the potential negative economic impact of a climate policy.\n\nSo, the most suitable label for this claim would be:\n\n4.1 - Climate policies (mitigation or adaptation) are harmful."
            },
            {
            "role": "user",
            "content": "But a problem surely arises when journals are run by IPCC insiders themselves."
            },
            {
            "role": "assistant",
            "content": "Given Claim: \"But a problem surely arises when journals are run by IPCC insiders themselves.\"\n\nThis comment questions the reliability and potential bias within the climate movement, specifically referring to journalists and the Intergovernmental Panel on Climate Change (IPCC).\n\nHence, as the claim argues there is some level of corruption within the climate movement, the most appropriate label for this claim would be :\n\n5.2 - Climate movement is unreliable/alarmist/corrupt"
            }, 
            {
            "role": "user",
            "content": f"{paragraph}"
            },
        ],
        temperature=0
        )

        final = ai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": "You have been provided analysis from a human labeler. Please extract solely the correct label in the form 0_0, ect. Only provide the main label the human determined.\n\n\n\n\n\n"
            },
            {
            "role": "user",
            "content": "Let's break down the provided claim: \"But a problem surely arises when journals are run by IPCC insiders themselves.\"\n\nThis statement suggests that there may be some issues or biases when journals related to climate change are run or controlled by those who are inside the Intergovernmental Panel on Climate Change (IPCC).\n\nThe main thrust of the claim is questioning the reliability or potential bias of climate-related science due to potential conflicts of interest.\n\nTherefore, the best label for this claim is:\n5.1 - Climate-related science is unreliable/uncertain/unsound (data, methods & models)"
            },
            {
            "role": "assistant",
            "content": "5_1"
            },
            {
            "role": "user",
            "content": "\nGiven Claim: \"NBCC/CRA study estimates Waxman-Markey would reduce employment by 2.3 million jobs in 2015\"\n\nThe primary focus of this claim is on the potential negative economic impact of a climate policy.\n\nSo, the most suitable label for this claim would be:\n\n4.1 - Climate policies (mitigation or adaptation) are harmful."
            },
            {
            "role": "assistant",
            "content": "4_1"
            },
            {
            "role": "user",
            "content": "Claim: \"Met Office Predicted A Warm Winter. Cheers Guys.\"\n\nTo determine the appropriate label for this claim, we need to understand its implication. The statement is pointing out a specific weather prediction (a warm winter) by the Met Office and seems to suggest that such predictions are either incorrect or dismissive. The sarcastic \"Cheers Guys\" suggests a level of mistrust in the prediction.\n\nFrom the provided labels, the following ones might be applicable:\n\n1.3 - Weather is cold/snowing: The claim could be implying that despite the prediction, the weather is cold or snowing. However, the claim does not directly state this, so it's a potential but not definite match.\n\n5.1 - Climate-related science is unreliable/uncertain/unsound (data, methods & models): This label could be inferred since the claim could be suggesting that the Met Office's prediction is unreliable.\n\nGiven the information provided in the claim, the best label seems to be:\n\n**5.1 - Climate-related science is unreliable/uncertain/unsound (data, methods & models)**\n\nThe claim appears to be challenging the reliability of climate predictions, which falls under the broader category of questioning the reliability of climate-related science."
            },
            {
            "role": "assistant",
            "content": "5_1"
            },
            {
            "role": "user",
            "content": f"{label_raw.choices[0].message.content}"
            }
        ],
        temperature=0
        )
        return final.choices[0].message.content
    elif extracted_response.choices[0].message.content == "NO":
        second_filter = ai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": "Let's think step by step. You are a labeler. \nPlease identify the paragraph as being \"orange\" if it contains any of the following points:\n-Is critical of common arguments for climate action or specific instances of climate action \n-Argues against traditional climate policy and emissions reduction\n-Mentions positive effects CO2 has\n-Mentions an example of high temperature in the past. \n\nLet's think step by step. "
            },
            {
            "role": "user",
            "content": "\"As Dan Kish, Senior Vice President of the Institute for Energy Research in Washington, D.C., observed in an interview I posted last March , the Obama administrations apparent intent is tomake heavily subsidized, unreliable and costly renewable energy programs they are pushing more cost-competitive. This is the Tonya Harding approach to energy break your opponents kneecap if you cant win fair and square.\""
            },
            {
            "role": "assistant",
            "content": "The paragraph is critical of a specific instance of climate action, namely the Obama administration's renewable energy programs. Therefore, it is \"orange\"."
            },
            {
            "role": "user",
            "content": "Death Valley : The summer of 1917 had 43 consecutive days with a high temperature of 120 F or above."
            },
            {
            "role": "assistant",
            "content": "The paragraph mentions an example of high temperature in the past. Therefore, it is \"orange\"."
            },
            {
            "role": "user",
            "content": "\"Regardless of atmospheric oxygen concentration, shoots of both mint and thyme exposed to elevated CO 2 produced substantially more biomass than shoots grown at 350 ppm CO 2 . At an ambient O 2 concentration of 21%, for example, CO 2 -enriched shoots of mint and thyme attained fresh weights that were 3.1- and 5.8-fold greater, respectively, than those reached by shoots exposed to ambient O 2 and CO 2 concentrations. What it means\""
            },
            {
            "role": "assistant",
            "content": "The paragraph mentions positive effects CO2 has on the growth of mint and thyme. Therefore, it is \"orange\"."
            },
            {
            "role": "user",
            "content": f"{paragraph}"
            }
        ],
        temperature=0
        )

        get_orange = ai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
            "role": "system",
            "content": "Your job is to extract a label from a human labeler's analysis. Please label the provided analysis and respond \"YES\" if the human identified the paragraph as \"orange\" and \"NO\" if not. "
            },
            {
            "role": "user",
            "content": "The paragraph mentions an example of high temperature in the past. Therefore, it is \"orange\"."
            },
            {
            "role": "assistant",
            "content": "YES"
            },
            {
            "role": "user",
            "content": "The paragraph mentions positive effects CO2 has on the growth of mint and thyme. Therefore, it is \"orange\"."
            },
            {
            "role": "assistant",
            "content": "YES"
            },
            {
            "role": "user",
            "content": "The paragraph mentions none of the criteria. Therefore, it isn't \"orange\"."
            },
            {
            "role": "assistant",
            "content": "NO"
            },
            {
            "role": "user",
            "content": f"{second_filter.choices[0].message.content}"
            }
        ],
        temperature=0
        )

        if get_orange.choices[0].message.content == "NO":
            return "0_0"
        else: 
            label_raw = ai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                "role": "system",
                "content": "You are a human labeler who labels climate denial claims.\\nHere are the labels:\\n1.1 - Ice/permafrost/snow cover isn’t melting\\n1.2 - We’re heading into an ice age/global cooling\\n1.3 - Weather is cold/snowing\\n1.4 - Climate hasn’t warmed/changed over the last (few) decade(s)\\n1.6 - Sea level rise is exaggerated/not accelerating\\n1.7 - Extreme weather isn’t increasing/has happened before/isn’t linked to climate change\\n2.1 - It’s natural cycles/variation\\n2.3 - There’s no evidence for greenhouse effect/carbon dioxide driving climate change\\n3.1 - Climate sensitivity is low/negative feedbacks reduce warming\\n3.2 - Species/plants/reefs aren’t showing climate impacts/are benefiting from climate change\\n3.3 - CO2 is beneficial/not a pollutant\\n4.1 - Climate policies (mitigation or adaptation) are harmful\\n4.2 - Climate policies are ineffective/flawed\\n4.4 - Clean energy technology/biofuels won’t work\\n4.5 - People need energy (e.g. from fossil fuels/nuclear)5.2 - Climate science or movement is unreliable, alarmist, or due to corruption\\n\\n \\n5.1 - Climate science is uncertain (methods or models)\\nPlease label the provided claim with the correct label. Ultimately, determine the single best label based on what the claim is mostly about. Let's think step by step."
                },
                {
                "role": "user",
                "content": "It makes you wonder what created all that CO2 millions of years ago."
                },
                {
                "role": "assistant",
                "content": "Alright, let's break down the claim: \"It makes you wonder what created all that CO2 millions of years ago.\"\n\nThis statement is referring to past events and implying that elevated levels of CO2 have occurred naturally in Earth's history without human intervention.\n\nGiven the provided labels:\n\n1.1 - 1.7 mainly pertain to observations about current climate phenomena (e.g., melting ice, extreme weather).\n2.1 - 2.3 are about the causes of climate change, with 2.1 highlighting natural cycles or variations, and 2.3 questioning the evidence for the greenhouse effect or CO2's role in driving climate change.\n3.1 - 3.3 discuss the direct impacts and perceptions of climate change, including the benefits or lack thereof of CO2.\n4.1 - 4.5 are about climate policies and energy choices.\n5.1 - 5.2 discuss the reliability of climate-related science and the motivations of the climate movement.\nGiven this breakdown, the claim appears to align most closely with:\n\n2.1 - It’s natural cycles/variation\n\nThe statement is suggesting that there were natural sources or processes in the Earth's history that led to high levels of CO2, which would fall under the category of natural cycles or variations."
                },
                {
                "role": "user",
                "content": "NBCC/CRA study estimates Waxman-Markey would reduce employment by 2.3 million jobs in 2015"
                },
                {
                "role": "assistant",
                "content": "\nChatGPT\nGiven Claim: \"NBCC/CRA study estimates Waxman-Markey would reduce employment by 2.3 million jobs in 2015\"\n\nThe primary focus of this claim is on the potential negative economic impact of a climate policy.\n\nSo, the most suitable label for this claim would be:\n\n4.1 - Climate policies (mitigation or adaptation) are harmful."
                },
                {
                "role": "user",
                "content": "But a problem surely arises when journals are run by IPCC insiders themselves."
                },
                {
                "role": "assistant",
                "content": "Given Claim: \"But a problem surely arises when journals are run by IPCC insiders themselves.\"\n\nThis comment questions the reliability and potential bias within the climate movement, specifically referring to journalists and the Intergovernmental Panel on Climate Change (IPCC).\n\nHence, as the claim argues there is some level of corruption within the climate movement, the most appropriate label for this claim would be :\n\n5.2 - Climate movement is unreliable/alarmist/corrupt"
                }, 
                {
                "role": "user",
                "content": f"{paragraph}"
                },
            ],
            temperature=0
            )

            final = ai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                "role": "system",
                "content": "You have been provided analysis from a human labeler. Please extract solely the correct label in the form 0_0, ect. Only provide the main label the human determined.\n\n\n\n\n\n"
                },
                {
                "role": "user",
                "content": "Let's break down the provided claim: \"But a problem surely arises when journals are run by IPCC insiders themselves.\"\n\nThis statement suggests that there may be some issues or biases when journals related to climate change are run or controlled by those who are inside the Intergovernmental Panel on Climate Change (IPCC).\n\nThe main thrust of the claim is questioning the reliability or potential bias of climate-related science due to potential conflicts of interest.\n\nTherefore, the best label for this claim is:\n5.1 - Climate-related science is unreliable/uncertain/unsound (data, methods & models)"
                },
                {
                "role": "assistant",
                "content": "5_1"
                },
                {
                "role": "user",
                "content": "\nGiven Claim: \"NBCC/CRA study estimates Waxman-Markey would reduce employment by 2.3 million jobs in 2015\"\n\nThe primary focus of this claim is on the potential negative economic impact of a climate policy.\n\nSo, the most suitable label for this claim would be:\n\n4.1 - Climate policies (mitigation or adaptation) are harmful."
                },
                {
                "role": "assistant",
                "content": "4_1"
                },
                {
                "role": "user",
                "content": "Claim: \"Met Office Predicted A Warm Winter. Cheers Guys.\"\n\nTo determine the appropriate label for this claim, we need to understand its implication. The statement is pointing out a specific weather prediction (a warm winter) by the Met Office and seems to suggest that such predictions are either incorrect or dismissive. The sarcastic \"Cheers Guys\" suggests a level of mistrust in the prediction.\n\nFrom the provided labels, the following ones might be applicable:\n\n1.3 - Weather is cold/snowing: The claim could be implying that despite the prediction, the weather is cold or snowing. However, the claim does not directly state this, so it's a potential but not definite match.\n\n5.1 - Climate-related science is unreliable/uncertain/unsound (data, methods & models): This label could be inferred since the claim could be suggesting that the Met Office's prediction is unreliable.\n\nGiven the information provided in the claim, the best label seems to be:\n\n**5.1 - Climate-related science is unreliable/uncertain/unsound (data, methods & models)**\n\nThe claim appears to be challenging the reliability of climate predictions, which falls under the broader category of questioning the reliability of climate-related science."
                },
                {
                "role": "assistant",
                "content": "5_1"
                },
                {
                "role": "user",
                "content": f"{label_raw.choices[0].message.content}"
                }
            ],
            temperature=0
            )
            return final.choices[0].message.content
    else:
        return "failed conversion"

def compute_scores(predictions, ground_truth):
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, average='macro', zero_division=1)
    recall = recall_score(ground_truth, predictions, average='macro', zero_division=1)
    
    return accuracy, precision, recall

def main():
    # 1. Read the CSV
    df = pd.read_csv('/Users/typham-swann/Desktop/CDL/sampled_output.csv')
    
    # Assuming your CSV has columns 'paragraph' and 'label'
    paragraphs = df['text'].tolist()
    correct_labels = df['claim'].tolist()
    
    # 2. Classify each paragraph
    predictions = []
    for index, p in enumerate(paragraphs):
        predictions.append(classify_paragraph(p))
        print(f"Labeled sample {index + 1}/{len(paragraphs)}")
    
    # 3. Compute accuracy
    accuracy, precision, recall = compute_scores(predictions, correct_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")

    # 4. Extract incorrect predictions and save them to a new CSV
    incorrect_data = {
        'paragraph': [],
        'predicted_label': [],
        'ground_truth': []
    }
    
    for para, predicted, ground_truth in zip(paragraphs, predictions, correct_labels):

        if predicted != ground_truth:
            incorrect_data['paragraph'].append(para)
            incorrect_data['predicted_label'].append(predicted)
            incorrect_data['ground_truth'].append(ground_truth)
    
    incorrect_df = pd.DataFrame(incorrect_data)
    incorrect_df.to_csv('incorrect_predictions.csv', index=False)
    print(f"Incorrect predictions saved to 'incorrect_predictions.csv'")


if __name__ == "__main__":
    main()


