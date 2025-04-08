"""Email evaluation dataset with ground truth classifications."""

# Dataset examples
email_input_1 = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi Lance,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

email_input_2 = {
    "author": "Marketing Team <marketing@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "New Company Newsletter Available",
    "email_thread": """Hello Lance,

The latest edition of our company newsletter is now available on the intranet. This month features articles on our Q2 results, upcoming team building activities, and employee spotlights.

Check it out when you have a chance!

Best regards,
Marketing Team""",
}

email_input_3 = {
    "author": "System Admin <sysadmin@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Scheduled maintenance - database downtime",
    "email_thread": """Hi Lance,

This is a reminder that we'll be performing scheduled maintenance on the production database tonight from 2AM to 4AM EST. During this time, all database services will be unavailable.

Please plan your work accordingly and ensure no critical deployments are scheduled during this window.

Thanks,
System Admin Team""",
}

email_input_4 = {
    "author": "Project Manager <pm@client.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Tax season let's schedule call",
    "email_thread": """Lance,

It's tax season again, and I wanted to schedule a call to discuss your tax planning strategies for this year. I have some suggestions that could potentially save you money.

Are you available sometime next week? Tuesday or Thursday afternoon would work best for me, for about 45 minutes.

Regards,
Project Manager""",
}

email_input_5 = {
    "author": "HR Department <hr@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Reminder: Submit your expense reports",
    "email_thread": """Hello Lance,

This is a friendly reminder that all expense reports for the previous month need to be submitted by this Friday. Please make sure to include all receipts and proper documentation.

If you have any questions about the submission process, feel free to reach out to the HR team.

Best regards,
HR Department""",
}

email_input_6 = {
    "author": "Conference Organizer <events@techconf.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Do you want to attend this conference?",
    "email_thread": """Hi Lance,

We're reaching out to invite you to TechConf 2025, happening May 15-17 in San Francisco. 

The conference features keynote speakers from major tech companies, workshops on AI and ML, and great networking opportunities. Early bird registration is available until April 30th.

Would you be interested in attending? We can also arrange for group discounts if other team members want to join.

Best regards,
Conference Organizers""",
}

email_input_7 = {
    "author": "Sarah Johnson <sarah.j@partner.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Can you review these docs before submission?",
    "email_thread": """Lance,

I've attached the final version of our proposal for the Henderson project. Could you please review the technical specifications section (pages 15-20) before we submit it to the client on Friday?

Your expertise would really help ensure we've covered all the necessary details.

Thanks in advance,
Sarah""",
}

email_input_8 = {
    "author": "Community Pool <info@cityrecreation.org>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Sign up daughter for swimming class",
    "email_thread": """Dear Lance,

Summer swimming registration is now open! Based on your daughter's participation last year, we wanted to let you know that intermediate level classes are available on Mondays and Wednesdays at 4PM or Tuesdays and Thursdays at 5PM.

Classes begin June 1st and run for 8 weeks. Space is limited, so early registration is recommended.

Please let us know if you'd like to reserve a spot.

Regards,
City Recreation Department""",
}

email_input_9 = {
    "author": "GitHub <notifications@github.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "PR #42: Comment from alex-dev",
    "email_thread": """Hey there!

alex-dev commented on your pull request #42 in langchain-ai/project:

> I've reviewed the changes and everything looks good. Just one small suggestion for the error handling in auth_controller.py. Maybe we should add a timeout parameter to prevent hanging requests?

View the comment: https://github.com/langchain-ai/project/pull/42#comment-12345

---
You're receiving this because you authored the thread.
Reply to this email directly, or view it on GitHub
""",
}

email_input_10 = {
    "author": "Team Lead <teamlead@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Quarterly planning meeting",
    "email_thread": """Hi Lance,

It's time for our quarterly planning session. I'd like to schedule a 90-minute meeting next week to discuss our roadmap for Q3.

Could you let me know your availability for Monday or Wednesday? Ideally sometime between 10AM and 3PM.

Looking forward to your input on the new feature priorities.

Best,
Team Lead""",
}

email_input_11 = {
    "author": "AWS Monitoring <no-reply@aws.amazon.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "System admin alert: Instance CPU utilization exceeds threshold",
    "email_thread": """ALERT: High CPU Utilization

The following EC2 instance has exceeded the CPU utilization threshold of 90% for more than 15 minutes:

Instance ID: i-0b2d3e4f5a6b7c8d9
Region: us-west-2
Current utilization: 95.3%

This message is automatically generated. Please do not reply.
""",
}

email_input_12 = {
    "author": "Client Success <success@vendor.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Your subscription will renew automatically",
    "email_thread": """Hello Lance,

This is a friendly reminder that your annual subscription to our Developer Pro plan will automatically renew on 04/15/2025.

Your payment method ending in **** 4567 will be charged $1,499.00.

If you would like to make any changes to your subscription, please visit your account settings or contact our support team before the renewal date.

Thank you for your continued business!

Client Success Team""",
}

email_input_13 = {
    "author": "Dr. Roberts <droberts@medical.org>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Annual checkup reminder",
    "email_thread": """Hello Lance,

This is a reminder that it's time for your annual checkup. Our records show that your last visit was approximately one year ago.

Please call our office at (555) 123-4567 to schedule an appointment at your earliest convenience.

Best regards,
Dr. Roberts' Office""",
}

email_input_14 = {
    "author": "Social Media Platform <notifications@social.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "5 people liked your post",
    "email_thread": """Hi Lance,

5 people liked your recent post about "Machine Learning Techniques for NLP"

See who liked your post and continue the conversation!

[View activity]

To unsubscribe from these notifications, adjust your settings here.
""",
}

email_input_15 = {
    "author": "Project Team <project@company.com>",
    "to": "Lance Martin <lance@company.com>",
    "subject": "Joint presentation next month",
    "email_thread": """Hi Lance,

The leadership team has asked us to prepare a joint presentation on our recent project successes for the all-hands meeting next month.

I've started putting together some slides and would appreciate your input on the technical architecture section. Could we schedule about 60 minutes sometime in the next week to collaborate on this?

I'm generally free on Tuesdays and Thursdays.

Thanks,
Project Team""",
}

# Triage outputs: "ignore", "notify", "respond"
triage_output_1 = "respond"
triage_output_2 = "ignore"
triage_output_3 = "notify"
triage_output_4 = "respond"
triage_output_5 = "notify"
triage_output_6 = "respond"
triage_output_7 = "respond"
triage_output_8 = "respond"
triage_output_9 = "notify"
triage_output_10 = "respond"
triage_output_11 = "notify"
triage_output_12 = "ignore"
triage_output_13 = "respond"
triage_output_14 = "ignore"
triage_output_15 = "respond"

# End-to-end response outputs (when applicable)
response_output_1 = "I've reviewed the API documentation and confirmed the missing endpoints. I'll add the /auth/refresh and /auth/validate endpoints to the documentation and send you the updated version."
response_output_2 = "No response is needed. For evaluation purposes, just confirm that the agent is to ignore."
response_output_3 = "No response is needed. For evaluation purposes, just confirm that the agent is to notify."
response_output_4 = "I set up some time to discuss tax planning strategies for 30 minutes. Looking forward to our call."
response_output_5 = "No response is needed. For evaluation purposes, just confirm that the agent is to notify."
response_output_6 = "Thank you for the invitation to TechConf 2025. I'm interested in attending and would like to take advantage of the early bird registration. Could you provide more details about the AI and ML workshops specifically?"
response_output_7 = "I'll review the technical specifications section of the Henderson project proposal (pages 15-20) and provide feedback by Thursday. Is there anything specific you'd like me to focus on?"
response_output_8 = "I'd like to register my daughter for the intermediate swimming class on Tuesdays and Thursdays at 5PM. Please let me know what information you need to complete the registration."
response_output_9 = "No response is needed. For evaluation purposes, just confirm that the agent is to notify."
response_output_10 = "I set up time for the quarterly planning meeting for 30 minutes."
response_output_11 = "No response is needed. For evaluation purposes, just confirm that the agent is to notify."
response_output_12 = "No response is needed. For evaluation purposes, just confirm that the agent is to ignore."
response_output_13 = "Thank you for the reminder about my annual checkup. I'll call your office this week to schedule an appointment."
response_output_14 = "No response is needed. For evaluation purposes, just confirm that the agent is to ignore."
response_output_15 = "I'd be happy to collaborate on the presentation. I scheduled a meeting for 30 minutes to discuss the technical architecture section. I'll review the slides you've prepared before then."

examples_triage = [
  {
      "inputs": {"email_input": email_input_1},
      "outputs": {"classification": triage_output_1},
  },
  {
      "inputs": {"email_input": email_input_2},
      "outputs": {"classification": triage_output_2},
  },
  {
      "inputs": {"email_input": email_input_3},
      "outputs": {"classification": triage_output_3},
  },
  {
      "inputs": {"email_input": email_input_4},
      "outputs": {"classification": triage_output_4},
  },
  {
      "inputs": {"email_input": email_input_5},
      "outputs": {"classification": triage_output_5},
  },
  {
      "inputs": {"email_input": email_input_6},
      "outputs": {"classification": triage_output_6},
  },
  {
      "inputs": {"email_input": email_input_7},
      "outputs": {"classification": triage_output_7},
  },
  {
      "inputs": {"email_input": email_input_8},
      "outputs": {"classification": triage_output_8},
  },
  {
      "inputs": {"email_input": email_input_9},
      "outputs": {"classification": triage_output_9},
  },
  {
      "inputs": {"email_input": email_input_10},
      "outputs": {"classification": triage_output_10},
  },
  {
      "inputs": {"email_input": email_input_11},
      "outputs": {"classification": triage_output_11},
  },
  {
      "inputs": {"email_input": email_input_12},
      "outputs": {"classification": triage_output_12},
  },
  {
      "inputs": {"email_input": email_input_13},
      "outputs": {"classification": triage_output_13},
  },
  {
      "inputs": {"email_input": email_input_14},
      "outputs": {"classification": triage_output_14},
  },
  {
      "inputs": {"email_input": email_input_15},
      "outputs": {"classification": triage_output_15},
  },
]

examples_response = [
  {
      "inputs": {"email_input": email_input_1},
      "outputs": {"response": response_output_1},
  },
  {
      "inputs": {"email_input": email_input_2},
      "outputs": {"response": response_output_2},
  },
  {
      "inputs": {"email_input": email_input_3},
      "outputs": {"response": response_output_3},
  },
  {
      "inputs": {"email_input": email_input_4},
      "outputs": {"response": response_output_4},
  },
  {
      "inputs": {"email_input": email_input_5},
      "outputs": {"response": response_output_5},
  },
  {
      "inputs": {"email_input": email_input_6},
      "outputs": {"response": response_output_6},
  },
  {
      "inputs": {"email_input": email_input_7},
      "outputs": {"response": response_output_7},
  },
  {
      "inputs": {"email_input": email_input_8},
      "outputs": {"response": response_output_8},
  },
  {
      "inputs": {"email_input": email_input_9},
      "outputs": {"response": response_output_9},
  },
  {
      "inputs": {"email_input": email_input_10},
      "outputs": {"response": response_output_10},
  },
  {
      "inputs": {"email_input": email_input_11},
      "outputs": {"response": response_output_11},
  },
  {
      "inputs": {"email_input": email_input_12},
      "outputs": {"response": response_output_12},
  },
  {
      "inputs": {"email_input": email_input_13},
      "outputs": {"response": response_output_13},
  },
  {
      "inputs": {"email_input": email_input_14},
      "outputs": {"response": response_output_14},
  },
  {
      "inputs": {"email_input": email_input_15},
      "outputs": {"response": response_output_15},
  },
]