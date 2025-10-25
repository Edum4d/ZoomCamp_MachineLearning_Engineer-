import pickle
input_file = 'pipeline_v1.bin'
def main():
    with open(input_file, 'rb') as f_in: 
        model = pickle.load(f_in)
    customer=    {
        "lead_source": "paid_ads",
        "number_of_courses_viewed": 2,
        "annual_income": 79276.0
    }

    y_pred = model.predict_proba(customer)[0, 1]
    print(y_pred)

if __name__ == "__main__":
    main()
