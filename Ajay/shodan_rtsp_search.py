import shodan

def search_rtsp_streams(api_key, query):
    try:
        # Create a Shodan API object
        api = shodan.Shodan(api_key)

        # Search Shodan
        results = api.search(query)
        for result in results['matches']:
            ip = result['ip_str']
            ports = result['ports']
            hostnames = result['hostnames']
            data = result['data']

            print(f"IP: {ip}")
            print(f"Ports: {ports}")
            print(f"Hostnames: {hostnames}")
            print(f"Data: {data}") 

    except shodan.APIError as e:
        print(f'Error: {e}')
        if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
            print(f'HTTP Error Code: {e.response.status_code}')
        else:
            print('No HTTP error code available.')


def main():
    api_key = 'J1zKMGxxXLYvbRqr5ZXoXVxItUXOVtz2'
    query = 'port:554 has_screenshot:true'
    search_rtsp_streams(api_key, query)

if __name__ == "__main__":
    main()
