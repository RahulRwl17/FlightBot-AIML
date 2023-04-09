class UserIntent:
    # create lists to hold queries and labels
    queries = []
    labels = []


    # load the intents dictionary
    intents = {

        "thank you": [
            "Thank you for your help.",
            "Thanks, that's all I needed.",
            "Appreciate your assistance.",
            "Thank you so much for your help, I really appreciate it!",
            "I just wanted to say thank you for all your assistance.",
            "Thanks a lot, you've been very helpful!"
        ],

        "flight_status": [
            "Can you tell me the status of flight DL456?",
            "What is the status of flight LH456 from Frankfurt to New York?",
            "Can you tell me if flight DL789 from Los Angeles to Miami is on time?",
            "Is flight AC123 from Toronto to London delayed or cancelled?",
            "What's the current status of flight AA123 from Dallas to New York?",
            "Is flight UA456 from San Francisco to Chicago running on time?",
            "Can you check the status of flight AC789 from Montreal to Vancouver?",     
            "Has there been any update on the status of flight LH456 from Frankfurt to New York?",
            "What's the delay status of flight DL789 from Los Angeles to Miami?",
            "Is there any information on the status of flight EK123 from Dubai to New York?"
        ],

        "flight_departure_time": [
            "When does flight BA456 depart from London Heathrow?",
            "What time is the departure of flight AF789 from Paris to Tokyo?",
            "At what time does flight LH123 leave from Frankfurt to Singapore?",
            "When is the departure time for flight DL123 from Los Angeles to New York?",
            "Can you tell me the time of departure for flight UA456 from Chicago to Boston?",
            "What time does flight AC789 leave from Toronto for London?",
            "At what time is flight EK321 departing from Dubai to New York?",
            "When is the scheduled departure time for flight LH456 from Frankfurt to New York?",
            "What time does flight AA789 leave from New York to Los Angeles?",
            "At what time does flight SQ123 depart from Singapore to Hong Kong?",    
            "When is the departure of flight QF321 from Sydney to Melbourne?",
            "Can you tell me the time of departure for flight KE456 from Seoul to Tokyo?",
            "What time does flight BA789 leave from London to New York?"
        ],

        "flight_gate_number": [
            "Can you give me the gate information for flight GH987?",
            "Which gate will flight DL456 from Atlanta to San Francisco depart from?",
            "Can you give me the gate information for flight UA789 from Chicago to Boston?",
            "What gate is flight EK123 from Dubai to New York leaving from?",
            "Which gate is flight BA456 from London Heathrow departing from?",
            "What's the gate number for flight CX789 from Hong Kong to Sydney?",
            "Can you tell me the gate number for my flight from Miami to New York?",
            "Where can I find the gate information for my flight from Tokyo to Los Angeles?",
            "What gate will flight LH123 from Frankfurt to Singapore arrive at?"
            ],

        "flight_arrival_time": [
           "What time will flight AA456 from New York to Los Angeles arrive?",
            "When is the arrival of flight AF789 from Paris to Tokyo?",
            "At what time does flight LH123 from Frankfurt to Singapore land?",
            "What is the expected arrival time for flight DL123 from Atlanta to Miami?",
            "When will flight BA456 from London Heathrow arrive in New York?",
            "What time does flight AA789 from Los Angeles to Chicago arrive?",
            "Can you tell me the expected arrival time for flight LH234 from Munich to Dubai?",
            "When is the estimated arrival time for flight EK567 from New York to Dubai?"
        ],
        "flight_number": [
            "What is the flight number for the flight from London to New York?",
            "Can you tell me the flight number for the flight from Miami to Los Angeles?",
            "What's the number of the flight from San Francisco to Chicago?",
            "What is the flight number for the United Airlines flight from Denver to Chicago?",
            "Can you tell me the flight number for the American Airlines flight from Dallas to Los Angeles?",
            "What is the flight number for the Delta flight from Atlanta to Boston?",
            "What's the flight number for the Air Canada flight from Vancouver to Toronto?",
            "Can you give me the flight number for the Southwest Airlines flight from Las Vegas to Denver?"
        ],
        "flight_airline": [
            "Which airline is operating the flight?",
            "What airline is my flight with?",
            "Can you tell me the name of the airline for my flight?",
            "Can you tell me the name of the airline for the flight AC123?",
            "What airline is operating the flight from Paris to London?",
            "Can you tell me the name of the airline for my flight from Los Angeles to New York?",
            "Which airline is flying from Sydney to Hong Kong?",
            "What airline is operating the flight from Frankfurt to New Delhi?",
            "Can you tell me the name of the airline for flight BA123 from London to Tokyo?"
        ],

        "flight_details": [
            "Can you show me the flights for American Airlines",
            "Can you provide me with information on flight PQR321?",
            "Can you show me flights from YYZ to YQB",
            "Can you tell me the flights details for flight from Canada to UK",
            "Tell me more about my flight and this is my flight number 1004.",
            "What is the best route to get from New York to Tokyo?",
            "What are the different routes for flying from Mumbai to London?",
            "Can you show me flights from UK to Canada on date April 7th, 2023?",
            "Can you provide me with information on flights from London to New York?",
            "What are the details of the flights from San Francisco to Chicago?",
            "Can you show me flights from Dubai to New York on April 1st, 2023?",
            "Can you provide me with more information about my flight?",
            "Can you tell me more about flights from New York to London?",
            "What are the different options for flying from Los Angeles to Tokyo?",
            "Can you show me flights from Singapore to Sydney?",
            "What are the flight details for flight AB789?",
        ]
    }



    # loop over each intent and add the queries and labels to the lists
    for intent, examples in intents.items():
        queries += examples
        labels += [intent] * len(examples)


    # convert the labels to a tensor
    label_dict = {label: i for i, label in enumerate(set(labels))}

