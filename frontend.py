import streamlit as st
import requests

st.set_page_config(layout='wide')

# Set the URL of the Flask app
flask_url = "http://127.0.0.1:5000/chat"


def filterResponse(res, filter_size=3):
    final_dict={}
    final_arr=[]
    for i in res:
        curr_brand = i["brand_name"]
        if(curr_brand not in final_dict):
            final_dict[curr_brand] = 1
            final_arr.append(i)

        elif(final_dict[curr_brand] < filter_size):
            final_dict[curr_brand] += 1
            final_arr.append(i)

    return final_arr


st.title("Chat with the Recommendation Model", anchor=False)
user_message = st.chat_input("Prompt: ")
if user_message:
    st.markdown("Prompt: " + user_message)

    # Making the request
    response = requests.post(flask_url, json={'prompt': user_message})

    st.markdown(f"Response Time: :green[{response.elapsed.total_seconds()}s]")
    st.divider()

    if response.status_code == 200:
        
        response = response.json()['response']['documents']
        response = filterResponse(response, filter_size=2)

        curr_columns=None
        for count, i in enumerate(response):  
            col_index = count % 4

            if col_index == 0:
                curr_container = st.container(border = False)
                curr_columns = curr_container.columns([4,4,4,4], gap = "large")

            c = curr_columns[col_index].container(border = True)
    
            c.subheader(i['brand_name'], divider = 'red', anchor=False)
            c.write(f"{i['details']}")
            c.write(f"ID: {i['offering_id']}")
            c.link_button("Visit", i['redemption_url'], use_container_width=True, type='primary')
    else:
        st.write("Error:", response.status_code)