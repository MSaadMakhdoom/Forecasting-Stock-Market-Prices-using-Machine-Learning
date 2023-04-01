import os
from psx import stocks
import datetime


def get_data():
    try:
   
        data = stocks(["FFC"], start=datetime.date(2000, 1, 1), end=datetime.date.today())
        
        # checking if the file exists
        if os.path.isfile("PSX.csv"):
            # appending the data to the existing file
            with open("PSX.csv", mode='a') as file:
                data.to_csv(file, index=True, header=False)
            print("Data appended to the file successfully!")
        else:
            # creating a new file and saving the data
            with open("PSX.csv", mode='w') as file:
                data.to_csv(file, index=True)
            print("New file created and data saved successfully!")
        
    except:
        print("Unable to get the data at the moment!")

        
        
    

def get_daily_data():
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=1)
        print("Start Date:", start_date)
        print("End Date:", end_date)
        data = stocks(["FFC"], start=start_date, end=end_date)
        print(data)
        print("Data saved successfully!")
    except:
        print("Unable to get the data at the moment!")



        
if __name__ == "__main__":
    get_daily_data()
