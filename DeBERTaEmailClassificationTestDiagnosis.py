import torch
from transformers import DebertaV2TokenizerFast, DebertaV2ForSequenceClassification
from collections import Counter

def load_model(model_path):
    model = DebertaV2ForSequenceClassification.from_pretrained(model_path)
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_path)
    return model, tokenizer

def classify_email(model, tokenizer, email_text):
    inputs = tokenizer(email_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = round(probabilities[0][predicted_class].item(), 2)
    categories = ['spam', 'query', 'complaint', 'suggestion']
    return categories[predicted_class], confidence

def run_diagnostics(model_path, test_data):
    model, tokenizer = load_model(model_path)
    
    print("Model Configuration:")
    print(model.config)
    
    label_counts = Counter(label for _, label in test_data)
    print("\nLabel distribution:", label_counts)
    
    prediction_counts = Counter()
    correct_confidences = []
    incorrect_confidences = []
    
    for email, true_label in test_data:
        prediction, confidence = classify_email(model, tokenizer, email)
        prediction_counts[prediction] += 1
        if prediction == true_label:
            correct_confidences.append(confidence)
        else:
            incorrect_confidences.append(confidence)
    
    print("\nPrediction distribution:", prediction_counts)
    
    print(f"\nAvg confidence for correct predictions: {sum(correct_confidences)/len(correct_confidences):.2f}")
    print(f"Avg confidence for incorrect predictions: {sum(incorrect_confidences)/len(incorrect_confidences):.2f}")
    
    print("\nMisclassified Examples:")
    for email, true_label in test_data:
        prediction, confidence = classify_email(model, tokenizer, email)
        if prediction != true_label:
            print(f"True: {true_label}, Predicted: {prediction}, Confidence: {confidence:.2f}")
            print(f"Email: {email[:100]}...\n")

# Test data
test_data = [
    ("I am writing to express my extreme disappointment and frustration regarding my recent order #12345. It has been over two weeks since I placed this order, and I still haven't received any update on its status or shipment. This delay is unacceptable, especially considering I paid for expedited shipping. I have tried reaching out to your customer service multiple times via email and phone, but I've received no response. This lack of communication is incredibly frustrating. I need this order urgently for an upcoming event, and your failure to deliver on time is causing significant inconvenience. I demand an immediate update on the status of my order, along with a full refund of my shipping costs. If I don't receive a satisfactory response within 24 hours, I will be forced to cancel my order and take my business elsewhere. I expect better service from a company of your reputation.", "complaint"),

    ("I recently purchased a high-end laptop from your company last month, and I have some questions regarding its warranty coverage. Could you please provide detailed information about what exactly is covered under the warranty? I'm particularly interested in knowing if accidental damage protection is included, and if so, what specific incidents are covered. Additionally, I'd like to know the duration of the warranty and if there are any options to extend it. In case I need to make a warranty claim, what is the process? Do I need to bring the laptop to a physical store, or can it be handled remotely? Are there any specific conditions that could void the warranty? Lastly, does the warranty cover international incidents, as I frequently travel for work? Your prompt and comprehensive response to these queries would be greatly appreciated.", "query"),

    ("I've been a loyal user of your software for several years now, and I have a suggestion that I believe would greatly enhance the user experience of your customer portal. Currently, the portal lacks a comprehensive dashboard that gives users an at-a-glance view of their account status, open support tickets, and current order statuses. I propose implementing a customizable dashboard that would allow users to see all of this critical information on a single page. This dashboard could include widgets for recent orders, shipping status, open support tickets, billing information, and perhaps even usage statistics for your software. The ability to customize this dashboard would be particularly useful, allowing users to prioritize the information most relevant to them. Additionally, integrating a notification system within this dashboard for important updates would significantly improve communication between your company and its customers. I believe this enhancement would streamline the user experience, reduce the number of support inquiries, and ultimately increase customer satisfaction. I'd be happy to discuss this suggestion further or provide any additional input if you find this idea worth pursuing.", "suggestion"),

    ("URGENT: Security Alert - Immediate Action Required! We have detected suspicious activity on your account that requires your immediate attention. Our security team has noticed multiple failed login attempts from an unrecognized IP address. To protect your account from unauthorized access, we need you to verify your identity and reset your password immediately. Click on the following link to begin the secure verification process: [suspicious link]. Failure to take action within the next 24 hours may result in a temporary suspension of your account for security reasons. Remember, your account security is our top priority. After resetting your password, we strongly recommend enabling two-factor authentication for an extra layer of protection. If you have any questions or concerns, please don't hesitate to contact our dedicated security team. Act now to keep your personal information and account safe!", "spam"),

    ("I hope this email finds you well. I'm reaching out because I'm having some difficulty understanding how to use the new feature that was recently added to your software in the latest update. Could you please provide a detailed explanation of its functionality and how to properly utilize it? Specifically, I'm unclear about how to access the feature, what its primary purpose is, and how it integrates with the existing tools in the software. Are there any specific settings I need to configure to optimize its performance? Additionally, I was wondering if there are any video tutorials or comprehensive guides available that demonstrate the feature in action. It would be incredibly helpful if you could walk me through a typical use case scenario. Lastly, are there any known limitations or potential issues with this new feature that I should be aware of? Your assistance in helping me understand and effectively use this new addition to your software would be greatly appreciated. Thank you in advance for your time and support.", "query"),

    ("I am writing to express my extreme dissatisfaction with the abysmal level of customer support I have experienced over the past week. Despite multiple attempts to reach your support team through various channels including phone, email, and your online chat system, I have received absolutely no response. This complete lack of communication is utterly unacceptable for a company that claims to value its customers. I have been a loyal client for over five years, and this is the first time I've encountered such a severe issue with your product. The software malfunction I'm experiencing has brought my work to a standstill, causing significant financial losses for my business. Your failure to address this problem in a timely manner is not only frustrating but also damaging to my operations. I demand an immediate response from a senior support representative who can resolve my issue promptly. Furthermore, I expect compensation for the downtime and losses I've incurred due to your negligence. If I don't receive a satisfactory resolution within the next 24 hours, I will have no choice but to terminate our business relationship and share my negative experience with others in my industry. This level of customer service is simply unacceptable.", "complaint"),

    ("I've been using your application for quite some time now, and I must say it has significantly improved my productivity. However, I believe there's room for an enhancement that could make the user experience even better. I'd like to suggest implementing a dark mode option in your app. This feature has become increasingly popular and for good reason. A dark mode would offer several benefits to your users. Firstly, it would reduce eye strain, especially for those who use the app for extended periods or in low-light conditions. This is particularly important for professionals who often work late into the night. Secondly, a dark mode could potentially save battery life on devices with OLED or AMOLED screens, which is a significant advantage for mobile users. Additionally, many users simply prefer the aesthetic of a dark interface. Implementing this feature could be done with a simple toggle in the settings menu, allowing users to switch between light and dark modes based on their preference or time of day. You could even consider adding an auto-switch feature that changes the mode based on the device's system settings or the time of day. I believe this addition would greatly enhance the overall user experience and potentially attract new users who prioritize this feature. I'd be happy to provide any further input or participate in beta testing if you decide to implement this suggestion.", "suggestion"),

    ("Attention Business Owners and Entrepreneurs! Are you ready to take your business to the next level? Don't miss out on this EXCLUSIVE offer designed specifically for ambitious professionals like you. For a limited time only, we're offering an unprecedented 75% discount on our Premium Business Solutions package. This all-in-one suite includes cutting-edge tools for project management, customer relationship management, financial forecasting, and much more. By subscribing today, you'll also receive: 1) Free access to our library of business masterclasses led by industry experts, 2) A personal business coach for the first month to help optimize your operations, 3) Lifetime priority customer support. But wait, there's more! The first 100 subscribers will also receive a free ticket to our annual Business Growth Summit, valued at $1,999. Don't let this opportunity slip away. Upgrade your business arsenal now and stay ahead of the competition. Click here to claim your discount before it expires in 24 hours. Remember, success waits for no one. Act fast and transform your business today!", "spam"),

("Hello, I'm contacting you regarding the invoice sent last month. It seems there's been a mistake in the total amount charged; I've been billed twice for the same service. Please look into this issue and revert with the corrected invoice. If possible, I would appreciate a phone call to discuss this matter in more detail. This kind of error is quite troubling for our financial planning, and prompt resolution would be greatly appreciated.", "complaint"), ("Good day, I am interested in upgrading my current service plan and am curious about any ongoing promotions or special offers that might be available. I’ve been a customer for several years and would like to understand what additional features or improvements have been made to the higher-tier plans. Can you provide detailed information on the differences between my current plan and the upgraded options? Thank you in advance for your help.", "query"), ("As a long-time user of your platform, I have noticed a trend where certain features become less responsive over time. My suggestion is to implement regular software updates that specifically focus on maintaining and improving performance, rather than only adding new features. This could help ensure that the user experience remains smooth, regardless of how long someone has been using your software. If needed, I am willing to participate in a beta testing group to provide real-time feedback on these updates.", "suggestion"), ("URGENT NOTICE: We've noticed unusual charges to your account from multiple locations. As a precaution, your account has been temporarily locked. To verify your identity and restore access, please follow the secure link attached. We understand the inconvenience this may cause and thank you for your quick attention to this serious matter. For your safety, do not share personal account details over email or phone with anyone claiming to represent us without verifying their identity first.", "spam"), ("I wanted to ask about the process for obtaining a refund for a purchase I made last week. The item does not function as advertised, and I have been unable to contact customer support through your listed phone numbers. Please advise on how I can return this product and get my money back as soon as possible. I have attached the purchase receipt and a video demonstrating the issue with the product for your reference.", "complaint"), ("Could you clarify the usage policy for your app’s API? I am developing a tool that integrates with your service and need to understand the rate limits and any potential costs associated with increased access requests. Any documentation or developer guides you can provide would be very helpful as I aim to ensure compliance and optimize the integration.", "query"), ("It would be beneficial for your mobile application to remember user preferences across updates. Currently, I have to reset my preferences every time the app updates, which is inconvenient and detracts from an otherwise seamless experience. Storing these settings in the user's profile could be a potential solution, enhancing user satisfaction and decreasing the need for repeated adjustments.", "suggestion"), ("Congratulations! You've been selected to receive a free trial of our exclusive cloud storage service, no payment information required at sign-up! Experience the best in data security and accessibility across all your devices. This offer is limited to the first 50 respondents, so act quickly to secure your spot and enjoy the benefits of our premium service completely free for the first month!", "spam"),

("We are reaching out to remind you that your subscription to our magazine is set to expire next month. We would hate for you to miss out on upcoming issues. Renew now and save 20% on your annual subscription. Stay informed, inspired, and entertained with our award-winning content. Renewal is quick and easy - just follow the link provided in this email.", "spam"), ("I have noticed a recurring error with my monthly billing statements. Charges from services that I did not authorize have been appearing for the past three months. This is unacceptable, and I need a detailed explanation along with immediate rectification. Additionally, I request a full audit of my account activities over the last quarter. I expect a prompt resolution to this issue.", "complaint"), ("Could you please update me on the status of the issue reported last week regarding network downtime? It has significantly impacted our operations and we need an urgent fix. Any temporary solutions would also be greatly appreciated while the main issue is being resolved. Please prioritize this matter and keep me informed of any developments.", "complaint"), ("I am currently exploring options for a new CRM system and came across your product online. Can you provide a comparison between your software and other leading systems in the market? Specifically, I am interested in features related to automation, scalability, and user interface. A detailed breakdown would help in making an informed decision.", "query"), ("As a frequent user of your photo editing software, I would love to see an enhancement in the layer management system. It would be incredibly useful to have the ability to group layers and search them by name. This feature would help streamline complex projects and improve workflow efficiency. Please consider this for your next update.", "suggestion"), ("This is your last chance to secure your seat at our exclusive webinar on digital marketing trends. Join industry experts as they discuss innovative strategies to enhance your online presence and drive growth. Click here to register for free and reserve your spot before it's too late. Don’t miss out on this opportunity to learn from the best in the business!", "spam"), ("I am writing to thank you for the prompt resolution of my previous complaint regarding the delivery issue. I am pleased with how your team handled the situation and the swift response in rectifying the error. Your commitment to customer satisfaction has not gone unnoticed, and I look forward to continuing my business with your company.", "complaint"), ("There has been a significant delay in the shipment of the products I ordered three weeks ago, and there has been no update from your team on the status. This delay is affecting our inventory levels and has the potential to impact our operations. Please provide an expedited update on the shipment and any possible solutions to mitigate this issue.", "complaint"), ("Can you explain how to access the advanced settings in your application? I've looked through the help documentation and forums but haven't found clear instructions. I’m particularly interested in customizing the workflow automation features to better fit our team's processes. A step-by-step guide would be very helpful.", "query"), ("It would be advantageous for your platform to integrate with third-party time tracking software. Many of your users, including myself, utilize external tools to monitor time spent on projects. Integrating this functionality directly into your platform could simplify workflows and attract users seeking an all-in-one solution.", "suggestion"),

("Dear Customer, thank you for your recent purchase from our store! We hope you are delighted with your new purchase. As part of our commitment to quality service, we are offering you a 10% discount on your next purchase with us. Just use the code THANKYOU10 at checkout. Offer expires soon, so don't miss out!", "spam"), ("I'm writing to express my dissatisfaction with the meal kit delivery. The last three deliveries have been late, and the quality of ingredients has significantly decreased. The produce was wilted, and the meat portions were smaller than what is advertised. This is not what I signed up for, and I expect either a compensation or an immediate improvement in service quality.", "complaint"), ("Hello, I was wondering if your software supports integration with ABC Accounting software? We are looking to streamline our workflow by integrating as many of our tools as possible. If not currently supported, could you indicate if this is in your development roadmap? Thank you for any information you can provide.", "query"), ("I would like to suggest an improvement for your mobile app. Adding biometric authentication for login would greatly enhance security and user convenience. Many apps are moving towards biometric options, and I think it would be beneficial for your app to incorporate this feature soon.", "suggestion"), ("Alert: Your account has been temporarily locked due to suspicious activity involving your credentials. To unlock your account, please verify your identity and secure your account by clicking on the link below. Do not share your password or security details with anyone. Security is our top priority!", "spam"), ("I need help resolving a billing error that occurred on my last invoice. The charge was for $250, whereas the agreed-upon monthly fee is $200. This seems to be an administrative error. Please rectify this at your earliest convenience and confirm once updated. Thank you for your attention to this matter.", "complaint"), ("Could you please let me know the process for exporting data from your system to a CSV file? I have been unable to find this option in the interface, and the help documentation does not seem to cover this functionality. Quick assistance would be greatly appreciated as I need this data urgently for an upcoming audit.", "query"), ("Here's a feature request for your team: It would be great if the software could provide more customizable reports. The current reports are useful, but they don’t allow us to modify the fields and filters extensively. Custom reports would help us to better analyze our data according to specific needs.", "suggestion"), ("Don’t miss out! Our annual summer sale starts next week! Prices will be slashed on all items—up to 50% off. Visit our website or any of our retail locations to take advantage of these fantastic deals before stock runs out. Save big on your favorite products!", "spam"), ("We have noted repeated instances of non-compliance with the project deadlines from your team. This has been affecting our operational efficiency and causing significant delays to related departments. Immediate corrective actions are required to address these issues. We expect a detailed plan of action within the next 48 hours.", "complaint"), ("Is there a way to customize notifications in your app? I'm receiving too many alerts that aren’t relevant to my daily tasks, and it’s becoming disruptive. I’d like to filter these notifications to only receive updates about critical issues and specific tasks I follow.", "query"), ("Integrating a dark mode into your application could significantly improve user experience, especially for users like myself who spend several hours on your platform. It's not only easier on the eyes but also more battery-friendly for mobile use. I hope to see this feature in a future update!", "suggestion"), ("Hurry up! Only a few spots left to join our exclusive webinar on the future of blockchain technology. Learn from top experts in the field and gain insights that could revolutionize how you manage transactions and data security. Register now to secure your spot and advance your knowledge!", "spam"),

 # Complaints
    ("I am extremely disappointed with the service I received during my last visit. The staff was unhelpful and rude, which is not what I expect from your establishment.", "complaint"),
    
    ("My recent experience with your product was frustrating. It did not work as advertised, and I would like a full refund.", "complaint"),
    
    ("I have been trying to resolve an issue with my account for over a week now, but I have received no assistance. This is unacceptable.", "complaint"),
    
    ("I ordered a product that arrived damaged, and I am not satisfied with how this has been handled. I expect a replacement immediately.", "complaint"),

    # Queries
    ("Can you clarify the terms of the service agreement? I'm unsure about the cancellation policy and any associated fees.", "query"),
    
    ("I would like to know how to reset my password. Could you provide detailed instructions?", "query"),
    
    ("What are the hours of operation for customer support? I need assistance but I'm not sure when to call.", "query"),
    
    ("Could you explain how to access my billing history on your website? I'm having trouble finding it.", "query"),

    # Suggestions
    ("I think adding a live chat feature would greatly improve customer support. It would allow customers to get immediate assistance.", "suggestion"),
    
    ("Implementing a loyalty program could enhance customer retention. Offering rewards for repeat purchases might encourage more sales.", "suggestion"),
    
    ("A mobile app would be beneficial for your service. Many users prefer using apps over websites for convenience.", "suggestion"),
    
    ("Consider providing more detailed product descriptions on your website. This would help customers make informed decisions.", "suggestion"),

    # Spam
    ("Congratulations! You've been selected to receive a free vacation package! Click here to claim your prize!", "spam"),
    
    ("Act now! You can earn thousands of dollars from home by participating in our survey program!", "spam"),
    
    ("Your account has been compromised! Click this link to verify your identity immediately!", "spam"),
    
    ("Limited time offer! Subscribe now and get 90% off our premium services for the first month!", "spam"),

    # Additional Complaints
    ("I am writing to express my frustration regarding the lack of response to my previous emails about my order status.", "complaint"),

    # Additional Queries
    ("What steps do I need to take to upgrade my account? I'm interested in additional features.", "query"),

    # Additional Suggestions
    ("Adding more payment options would make it easier for customers to complete their purchases. Consider including PayPal or Apple Pay.", "suggestion"),

    # Additional Spam
    ("You've won a gift card worth $1000! Click here to claim it before it expires!", "spam"),


 # Complaints
    ("The product I received was not as described on your website. I expected a different color and model. I would like to return it for a full refund.", "complaint"),
    
    ("I am very unhappy with the service I received from your support team. They were unhelpful and did not resolve my issue.", "complaint"),
    
    ("My internet connection has been unstable for weeks, and despite my numerous calls to customer support, no one has resolved the issue.", "complaint"),
    
    ("I have been charged incorrectly for my last bill. Please investigate this matter and correct the billing error.", "complaint"),

    # Queries
    ("Could you explain how to cancel my subscription? I want to ensure that I follow the correct procedure.", "query"),
    
    ("What are the available payment methods for your services? I want to know if you accept PayPal.", "query"),
    
    ("Can you provide details about your data privacy policy? I'm concerned about how my information is being used.", "query"),
    
    ("How can I update my account information? I need to change my email address and phone number.", "query"),

    # Suggestions
    ("I think it would be beneficial to have a feedback feature within the app. This would allow users to report issues directly.", "suggestion"),
    
    ("Adding a tutorial section in the app could help new users understand its features better. This would improve user satisfaction.", "suggestion"),
    
    ("Consider implementing a referral program to encourage existing users to bring in new customers. This could boost your user base significantly.", "suggestion"),
    
    ("It would be great if you could include a dark theme option in your app settings. Many users prefer this feature for better usability.", "suggestion"),

    # Spam
    ("Congratulations! You've been selected for an amazing prize! Claim your reward now by clicking this link!", "spam"),
    
    ("You have a pending payment that needs your immediate attention! Click here to resolve this issue now!", "spam"),
    
    ("Don't miss out on our exclusive offer! Sign up today and get 90% off your first purchase!", "spam"),
    
    ("Your account has been compromised! Verify your identity immediately by clicking this link!", "spam"),

 # Complaints
    ("I am writing to express my frustration with the delay in receiving my refund for order #54321. It has been over a month since I initiated the return process, and I have yet to see any progress. This is unacceptable.", "complaint"),
    
    ("The product I received was defective. I would like to request a replacement or a full refund as soon as possible.", "complaint"),
    
    ("I have been trying to reach your customer support regarding my subscription cancellation for weeks now. I have not received any response, and I am still being charged.", "complaint"),
    
    ("The quality of the service provided during my last interaction was below standard. I expected better from your team.", "complaint"),
    
    ("I am extremely unhappy with the recent changes made to your app. The new layout is confusing and not user-friendly.", "complaint"),

    # Queries
    ("Can you provide me with information about the upcoming features in your next software update? I'm eager to know what improvements are being made.", "query"),
    
    ("What is the procedure for escalating an issue if I am not satisfied with the initial response from customer support?", "query"),
    
    ("Could you clarify the differences between the various subscription plans you offer? I'm trying to decide which one is best for me.", "query"),
    
    ("How can I access my account settings? I need to update my personal information but can't find where to do it.", "query"),
    
    ("Is there a way to track my order status online? I would like to know when my package will arrive.", "query"),

    # Suggestions
    ("I suggest adding a feature that allows users to customize notifications based on their preferences. This would enhance user engagement.", "suggestion"),
    
    ("Implementing a chatbot for customer service inquiries could significantly reduce response times and improve user satisfaction.", "suggestion"),
    
    ("Consider offering a free trial period for new users. This could encourage more sign-ups and allow potential customers to experience your service firsthand.", "suggestion"),
    
    ("A community forum where users can share tips and experiences would be a great addition. It could foster user engagement and support.", "suggestion"),
    
    ("Adding a search function within the app would help users find specific features or information more quickly. This would improve overall usability.", "suggestion"),

    # Spam
    ("Congratulations! You've won an all-expenses-paid trip! Click here to claim your prize now!", "spam"),
    
    ("Urgent: Your account will be suspended unless you verify your information immediately!", "spam"),
    
    ("You have been selected for an exclusive offer! Sign up now and receive 80% off our premium services!", "spam"),
    
    ("Don't miss out on this limited-time opportunity! Join our program and earn money from home!", "spam"),
    
    ("Your invoice is attached! Please review and pay immediately to avoid penalties!", "spam"),

    # Additional Complaints
    ("I am disappointed with the lack of updates regarding my service request. It has been over two weeks without any communication.", "complaint"),

    # Additional Queries
    ("What are your policies regarding data privacy? I'm concerned about how my personal information is handled.", "query"),

    # Additional Suggestions
    ("It would be helpful if you could include a FAQ section on your website. This could address common questions and reduce support inquiries.", "suggestion"),

    # Additional Spam
    ("You've been chosen for a special reward! Claim your gift card now by clicking this link!", "spam")

]

# Model path
model_path = "./deberta_complaint_included_same_spam_ham_improved"

# Run diagnostics
print("Starting model diagnostics...")
run_diagnostics(model_path, test_data)

print("\nDiagnostics completed.")